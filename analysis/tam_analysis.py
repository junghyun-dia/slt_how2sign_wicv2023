"""
tam_analysis.py

slt_how2sign_wicv2023's Temporal Attention Map (TAM) extraction, analogous
to SpaMo's / vtamo's analysis/tam_analysis.py: for each test sentence,
teacher-force the model's own already-generated prediction (from
run_inference.py's predictions_original.csv) back through the model and
inspect how concentrated ("peaked") vs. spread ("diffuse") the decoder's
cross-attention is over the source (I3D feature) positions.

*** DRAFT / UNTESTED: no trained checkpoint exists yet. Only checked with
`python -m py_compile`. The cross-attention extraction path below is read
directly off this repo's fairseq/modules/transformer_layer.py +
fairseq/models/transformer/transformer_decoder.py (see derivation below);
everything else (loss formulation, decode chain) is best-effort. ***

How cross-attention is obtained (read, not guessed)
----------------------------------------------------
fairseq.models.transformer.transformer_decoder.TransformerDecoderBase.
extract_features_scriptable loops over decoder layers and, for
`idx == alignment_layer` (default: the LAST layer, `self.num_layers - 1`,
when alignment_layer=None), calls the layer with `need_attn=True,
need_head_weights=True` UNCONDITIONALLY, not gated behind any extra flag
the caller must opt into. That layer call (fairseq/modules/
transformer_layer.py, TransformerDecoderLayerBase.forward) runs
`self.encoder_attn(..., need_weights=True, need_head_weights=True)`, a
stock MultiheadAttention whose forward (fairseq/modules/
multihead_attention.py) returns attn_weights of shape
[num_heads, bsz, tgt_len, src_len] when need_head_weights=True. Back in
extract_features_scriptable this is averaged over heads
(`attn = attn.mean(dim=0)`) to shape [bsz, tgt_len, src_len] and returned
as `extra["attn"][0]`.

Concretely: Sign2TextTransformerModel.forward() (fairseq/models/
sign_to_text/sign2text_transformer.py) is just
`self.decoder(prev_output_tokens=..., encoder_out=self.encoder(...))`, and
TransformerDecoderBase.forward() returns (logits, extra) where
`extra["attn"][0]` IS this last-layer, head-averaged cross-attention --
already exposed by an ordinary forward pass, no output_attentions kwarg,
no hooks. This is the "need_attn-style API" the task brief said to check
for, used here exactly as read (no invented API).

Position mapping: exact, NOT approximate (unlike SpaMo)
---------------------------------------------------------
SpaMo needs a nontrivial receptive-field calculation because its
TemporalConv subsamples the visual token sequence. Sign2TextTransformerEncoder
does NOT subsample at all: forward() is just feat_proj(src_tokens) (a
per-timestep nn.Linear) + positional embedding + a stack of ordinary
(non-strided, non-pooling) TransformerEncoderLayers. Encoder output
position j is therefore EXACTLY input row j -- i.e. row j of the
precomputed per-sentence I3D .npy feature array loaded by
SignFeatsDataset.__getitem__. So "attention-source-index -> I3D feature
index" is the identity map: no arithmetic, no rounding, no verification
step needed.

What IS an approximation -- and is NOT attempted here
--------------------------------------------------------
"I3D feature index -> real video frame/time" is a separate mapping this
script deliberately does NOT compute. That would require the temporal
window/stride the upstream I3D backbone used to produce the *per-video*
feature matrix that examples/sign_language/scripts/i3d_formatting.py later
slices into per-sentence .npy files (that script only shows the
per-sentence slicing -- start = int(START_REALIGNED*fps) - 8 -- against a
full-video feature matrix loaded from an external GPI-cluster path,
".../featurize-How2Sign_c1887_m_d3_prebobsl-v0-stride0.0625/..."; the
actual I3D extraction code that produced THAT matrix, and hence its
window/stride, is not present anywhere in this repository). Rather than
guess a frame conversion from the ambiguous "stride0.0625" name, this
script reports everything in feature-index units (peak_feature_idx /
relative position within the sentence's own feature sequence) and flags
this explicitly instead of fabricating a frame/time axis.

Outputs (under analysis/predictions/):
    tam_stats.csv    -- per-sentence: sentence_id, n_tokens, mean_entropy, mean_peak_relpos
    tam_summary.csv  -- corpus-level aggregate stats
    tam_full.json    -- all sentences' full per-token attention rows (token, attn over
                         real_src_len positions, peak_feature_idx)

Usage (pick GPU 4/5/6/7; always nohup -- see project memory):
    conda activate <sign2text env>
    cd external/slt_how2sign_wicv2023
    CUDA_VISIBLE_DEVICES=4 python analysis/tam_analysis.py \
        --ckpt /path/to/checkpoint_best.pt \
        --data-dir /mnt/aix22303/data/how2sign/How2Sign/prepared/wicv23/i3d_features \
        --split cvpr23.fairseq.i3d.test.how2sign
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from fairseq import checkpoint_utils  # noqa: E402

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(ANALYSIS_DIR, "predictions")
PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "predictions_original.csv")

DEFAULT_TEST_SPLIT = "cvpr23.fairseq.i3d.test.how2sign"


def load_model_and_task(ckpt_path, data_dir, spm_model, device):
    arg_overrides = {"data": data_dir}
    if spm_model:
        arg_overrides["sentencepiece_model"] = spm_model
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [ckpt_path], arg_overrides=arg_overrides,
    )
    model = models[0]
    model.to(device)
    model.eval()
    return model, task


def encode_target(task, text, device):
    """Re-tokenize `text` (the model's own decoded prediction) the same way
    SignToTextTask.load_dataset's process_label_fn does, and build the
    (prev_output_tokens, target) pair the same way AddTargetDataset.collater
    does: prev_output_tokens = eos ++ tokens, target = tokens ++ eos. Row t
    of the decoder's cross-attention is therefore exactly the attention
    used to predict target[t]."""
    label = text.lower() if task.cfg.pre_tokenizer == "moses" else text
    bpe_encoded = task.bpe_tokenizer.encode(label)
    token_ids = task.target_dictionary.encode_line(
        bpe_encoded, append_eos=False, add_if_not_exist=False,
    ).long()
    eos = torch.LongTensor([task.target_dictionary.eos()])
    prev_output_tokens = torch.cat([eos, token_ids]).unsqueeze(0).to(device)
    target = torch.cat([token_ids, eos]).unsqueeze(0).to(device)
    return prev_output_tokens, target


def token_str(task, idx):
    return task.target_dictionary[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--spm-model", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.ckpt} on {device}...")
    model, task = load_model_and_task(args.ckpt, args.data_dir, args.spm_model, device)

    print(f"Loading dataset split '{args.split}'...")
    task.load_dataset(args.split)
    dataset = task.datasets[args.split]
    # SignFeatsDataset.ids is populated at construction time (from the tsv
    # manifest), so this is a cheap way to map vid_id -> integer dataset
    # index without triggering a .npy load per sentence.
    underlying = dataset.dataset  # AddTargetDataset -> SignFeatsDataset
    id_to_idx = {vid_id: i for i, vid_id in enumerate(underlying.ids)}

    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"{PREDICTIONS_CSV} not found -- run run_inference.py first to produce it."
        )
    preds_df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Running TAM extraction over {len(preds_df)} test sentences...")

    rows = []
    full_examples = []
    with torch.no_grad():
        for n_done, rec in enumerate(preds_df.itertuples(), 1):
            sid = rec.SENTENCE_ID
            if sid not in id_to_idx:
                print(f"  WARNING: {sid} not found in dataset, skipping")
                continue
            idx = id_to_idx[sid]
            sample = dataset[idx]  # {"id", "vid_id", "source": FloatTensor[T, feat_dim], "label"}
            src = sample["source"]
            if src.shape[0] == 0:
                continue
            real_src_len = src.shape[0]

            src_tokens = src.unsqueeze(0).float().to(device)  # [1, T, feat_dim]
            encoder_padding_mask = torch.zeros(1, real_src_len, dtype=torch.bool, device=device)

            pred_text = str(rec.prediction)
            prev_output_tokens, target = encode_target(task, pred_text, device)

            logits, extra = model(
                src_tokens=src_tokens,
                encoder_padding_mask=encoder_padding_mask,
                prev_output_tokens=prev_output_tokens,
            )
            # extra["attn"][0]: [1, tgt_len, src_len], last decoder layer,
            # already averaged over heads (see module docstring above).
            attn = extra["attn"][0]
            if attn is None:
                print(f"  WARNING: {sid} produced no attention (attn=None), skipping")
                continue
            attn = attn[0].float().cpu().numpy()  # [tgt_len, src_len]
            attn = attn[:, :real_src_len]
            attn = attn / attn.sum(axis=1, keepdims=True).clip(min=1e-8)

            target_ids = target[0].tolist()
            max_entropy = np.log(real_src_len) if real_src_len > 1 else 1.0

            entropies, peak_relpos, token_rows = [], [], []
            pad_idx = task.target_dictionary.pad()
            for t_idx, tok_id in enumerate(target_ids):
                if tok_id == pad_idx or t_idx >= attn.shape[0]:
                    continue
                tok = token_str(task, tok_id)
                row = attn[t_idx]
                ent = -np.sum(row * np.log(row.clip(min=1e-12)))
                entropies.append(ent / max_entropy)
                peak_j = int(np.argmax(row))
                peak_relpos.append(peak_j / max(real_src_len - 1, 1))
                token_rows.append(dict(
                    token=tok,
                    attn=[round(v, 4) for v in row.tolist()],
                    peak_feature_idx=peak_j,
                ))

            if not entropies:
                continue
            rows.append(dict(
                sentence_id=sid, n_tokens=len(entropies),
                mean_entropy=float(np.mean(entropies)),
                mean_peak_relpos=float(np.mean(peak_relpos)),
            ))
            full_examples.append(dict(
                sentence_id=sid, gt=rec.GT, prediction=pred_text,
                real_src_len=real_src_len, tokens=token_rows,
            ))

            if n_done % 25 == 0 or n_done == len(preds_df):
                print(f"  [{n_done}/{len(preds_df)}] last={sid}", flush=True)

    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join(PREDICTIONS_DIR, "tam_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"wrote {stats_path} ({len(stats_df)} sentences)")

    full_path = os.path.join(PREDICTIONS_DIR, "tam_full.json")
    with open(full_path, "w") as f:
        json.dump(full_examples, f)
    print(f"wrote {full_path} ({len(full_examples)} sentences, full per-token attention)")

    summary = dict(
        n_sentences=len(stats_df),
        mean_entropy=stats_df["mean_entropy"].mean() if len(stats_df) else float("nan"),
        median_entropy=stats_df["mean_entropy"].median() if len(stats_df) else float("nan"),
        mean_peak_relpos=stats_df["mean_peak_relpos"].mean() if len(stats_df) else float("nan"),
    )
    summary_path = os.path.join(PREDICTIONS_DIR, "tam_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print("\n=== TAM summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {summary_path}")
    print(
        "NOTE: unlike SpaMo, there is no region-masking-vs-peakiness comparison here "
        "(no masked-feature-extraction analysis exists yet for this model). "
        "Run gradcam_analysis.py next for a TAM-vs-GradCAM cross-check instead."
    )


if __name__ == "__main__":
    main()
