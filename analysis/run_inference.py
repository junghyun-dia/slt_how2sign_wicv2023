"""
run_inference.py

slt_how2sign_wicv2023's equivalent of SpaMo's / vtamo's analysis/run_inference.py:
loads a trained sign2text_transformer (fairseq task `sign_to_text`) checkpoint
and runs beam-search generation over a test split, writing a predictions CSV
with the same column schema SpaMo uses (SENTENCE_ID, GT, prediction, bleu,
chrf) so downstream tooling (add_semantic_similarity.py-style scripts,
region-masking-style comparisons, etc.) can treat all three models uniformly.

*** DRAFT / UNTESTED: no trained checkpoint exists yet for this model as of
writing. This script has only been checked with `python -m py_compile`; it
has NOT been run end-to-end. Treat argument defaults, dataset-split naming,
and the decode() detokenization chain as best-effort until verified against
a real checkpoint + data directory. ***

Bypass pattern
--------------
Unlike SpaMo (a PyTorch Lightning model we drive directly through
get_inputs()/shared_step()), this repo is a fairseq fork, so the natural
"bypass the CLI" story is different: fairseq-generate / fairseq-train are
thin wrappers around (a) `fairseq.tasks.FairseqTask.setup_task` +
`task.load_dataset(split)` to build the data pipeline, and (b)
`fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt], ...)` to
rebuild the exact task+model the checkpoint was trained with and load its
weights. This is fairseq's own standard, documented API for programmatic
checkpoint loading (used internally by fairseq-generate/interactive/eval
scripts) -- not something invented for this script.

The one existing local-modification file in this repo,
examples/sign_language/scripts/analyze_fairseq_generate.py, already
demonstrates the first half of this (directly instantiating
SignToTextConfig + SignToTextTask.setup_task + task.load_dataset(split) to
get at the raw references/vid_ids without going through fairseq-generate's
text output). It does NOT demonstrate checkpoint loading (it only
post-processes an already-produced fairseq-generate .out log file) -- that
part below uses fairseq's checkpoint_utils.load_model_ensemble_and_task
directly, passing arg_overrides={"data": ...} the same way fairseq-generate
does via its --path/--data CLI flags.

Because load_model_ensemble_and_task(task=None) rebuilds the task from the
*checkpoint's own saved cfg* (cfg.task, cfg.bpe, cfg.model, ...), we don't
need to separately reconstruct SignToTextConfig by hand: we only override
the `data` (and optionally `sentencepiece_model`) fields, in case the
checkpoint's absolute training-machine paths don't exist on this machine.

Generation vs. teacher-forcing
-------------------------------
This script does full beam-search generation (task.inference_step), which
is what a normal "predictions CSV" should report (matches SpaMo's
run_inference.py: `model.shared_step(inputs, "test", ...)` internally calls
the T5 model's own .generate()-based greedy/beam decoding). TAM/GradCAM
analysis then *teacher-forces* the model's own generated prediction back
through a single forward pass to extract attention -- see tam_analysis.py /
gradcam_analysis.py, which both read this script's predictions_original.csv
as their input.

Outputs (under analysis/predictions/):
    predictions_original.csv   (SENTENCE_ID, GT, prediction, bleu, chrf)

Resumable at the row-count level (skips if the CSV already has all rows for
the target split), same convention as SpaMo's run_inference.py.

Usage (pick GPU 4/5/6/7 -- never 0-3 -- once a checkpoint exists; always
nohup for anything beyond a quick sanity check -- see project memory):
    conda activate <sign2text env>
    cd external/slt_how2sign_wicv2023
    CUDA_VISIBLE_DEVICES=4 python analysis/run_inference.py \
        --ckpt /path/to/checkpoint_best.pt \
        --data-dir /mnt/aix22303/data/how2sign/How2Sign/prepared/wicv23/i3d_features \
        --split cvpr23.fairseq.i3d.test.how2sign
"""
import argparse
import csv
import os
import sys
import time

import torch
from sacrebleu.metrics import BLEU, CHRF

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from fairseq import checkpoint_utils, utils  # noqa: E402

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "predictions")

# Matches the manifest-name convention used throughout this repo, e.g.
# examples/sign_language/scripts/analyze_fairseq_generate.py's
# f"cvpr23.fairseq.i3d.{partition}.how2sign" and the train/valid_subset
# values in examples/sign_language/config/wicv_cvpr23/i3d_best/*.yaml.
# ASSUMPTION: adjust via --split if the actual trained checkpoint used a
# differently-named test manifest.
DEFAULT_TEST_SPLIT = "cvpr23.fairseq.i3d.test.how2sign"

# Try to strip BPE + moses-detokenize + truecase a generated/reference token
# sequence back into plain text, mirroring SignToTextTask.valid_step's local
# `decode()` closure (fairseq/tasks/sign_to_text.py) as closely as possible
# without depending on task.cfg.eval_bleu being set (valid_step only builds
# self.moses_detok when eval_bleu=True, which needn't be true for a bare
# eval-only task instance here).
try:
    import truecase
except ImportError:  # pragma: no cover - draft-time environment may differ
    truecase = None
try:
    from sacremoses import MosesDetokenizer
except ImportError:  # pragma: no cover
    MosesDetokenizer = None


def build_decode_fn(task):
    moses_detok = MosesDetokenizer(lang="en") if MosesDetokenizer is not None else None

    def decode(toks):
        if hasattr(task.sequence_generator, "symbols_to_strip_from_output"):
            to_ignore = task.sequence_generator.symbols_to_strip_from_output
        else:
            to_ignore = {task.sequence_generator.eos}
        s = task.tgt_dict.string(toks.int().cpu(), escape_unk=True, extra_symbols_to_ignore=to_ignore)
        if task.bpe_tokenizer:
            s = task.bpe_tokenizer.decode(s)
        # SignToTextTask.valid_step truecases via `truecase.get_true_case(s)`
        # whenever pre_tokenizer == 'moses' (it does NOT call
        # moses_detok.detokenize -- that call is present but commented out in
        # the original valid_step too; kept consistent with that behavior).
        if task.cfg.pre_tokenizer == "moses" and truecase is not None:
            s = truecase.get_true_case(s)
        return s

    return decode


def load_model_and_task(ckpt_path, data_dir, spm_model, device):
    arg_overrides = {"data": data_dir}
    if spm_model:
        # SignToTextConfig.bpe_sentencepiece_model is `II("bpe.sentencepiece_model")`,
        # i.e. an interpolation into the separate `bpe` sub-config; fairseq's
        # overwrite_args_by_name (used internally by load_checkpoint_to_cpu)
        # matches override keys by field name anywhere in the nested cfg, so
        # this should reach `cfg.bpe.sentencepiece_model`. NOT verified against
        # a real checkpoint -- if it doesn't take effect, override
        # `cfg.bpe.sentencepiece_model` directly after loading instead.
        arg_overrides["sentencepiece_model"] = spm_model

    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ckpt_path], arg_overrides=arg_overrides,
    )
    model = models[0]
    model.to(device)
    model.eval()
    return model, task, saved_cfg


def already_done(csv_path, n_expected):
    if not os.path.exists(csv_path):
        return False
    with open(csv_path) as f:
        n = sum(1 for _ in f) - 1  # minus header
    return n >= n_expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to a trained sign2text_transformer .pt checkpoint")
    parser.add_argument("--data-dir", required=True, help="Directory containing <split>.tsv manifests + vocab")
    parser.add_argument("--split", default=DEFAULT_TEST_SPLIT, help="Manifest name (no .tsv) to run inference over")
    parser.add_argument("--spm-model", default=None, help="Optional override path for the sentencepiece model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam", type=int, default=None, help="Override beam size (default: use checkpoint's own eval_gen_config)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None, help="Output CSV path (default: analysis/predictions/predictions_original.csv)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.ckpt} on {device}...")
    model, task, saved_cfg = load_model_and_task(args.ckpt, args.data_dir, args.spm_model, device)

    if args.beam is not None:
        from fairseq.dataclass.configs import GenerationConfig
        task.sequence_generator = task.build_generator([model], GenerationConfig(beam=args.beam))

    out_csv = args.output or os.path.join(OUTPUT_DIR, "predictions_original.csv")

    print(f"Loading dataset split '{args.split}' from {args.data_dir}...")
    task.load_dataset(args.split)
    dataset = task.datasets[args.split]
    n_expected = len(dataset)
    if already_done(out_csv, n_expected):
        print(f"already done ({out_csv}), skipping")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=dataset.collater,
    )

    decode = build_decode_fn(task)

    print(f"Running generation over {len(dataset)} sentences in '{args.split}'...")
    t0 = time.time()
    ids, gts, preds = [], [], []
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            if len(sample) == 0:
                continue  # SignFeatsDataset.collater returns {} if a batch is entirely all-padding
            sample = utils.move_to_cuda(sample, device=device) if device.type == "cuda" else sample
            gen_out = task.inference_step(task.sequence_generator, [model], sample, prefix_tokens=None)
            for i, sample_id in enumerate(sample["id"].tolist()):
                vid_id = dataset[sample_id]["vid_id"]
                gt = dataset.get_label(sample_id)
                pred_tokens = gen_out[i][0]["tokens"].int().cpu()
                pred = decode(pred_tokens)
                ids.append(vid_id)
                gts.append(gt)
                preds.append(pred)
            if (batch_idx + 1) % 10 == 0:
                print(f"  [{len(ids)}/{n_expected}]", flush=True)
    elapsed = time.time() - t0
    print(f"generation done in {elapsed:.0f}s ({len(ids)} sentences)")

    bleu_scorer = BLEU(effective_order=True)
    chrf_scorer = CHRF()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SENTENCE_ID", "GT", "prediction", "bleu", "chrf"])
        for sid, gt, pred in zip(ids, gts, preds):
            bleu = bleu_scorer.sentence_score(pred, [gt]).score
            chrf = chrf_scorer.sentence_score(pred, [gt]).score
            writer.writerow([sid, gt, pred, f"{bleu:.4f}", f"{chrf:.4f}"])

    if preds:
        corpus_bleu = BLEU().corpus_score(preds, [gts]).score
        print(f"corpus BLEU-4 = {corpus_bleu:.2f} (sanity check against training-time eval_bleu logs)")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
