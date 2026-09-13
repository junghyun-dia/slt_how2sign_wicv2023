"""
gradcam_analysis.py

slt_how2sign_wicv2023's transformer-adapted Grad-CAM (attention x
gradient), analogous to SpaMo's / vtamo's analysis/gradcam_analysis.py.
Same setup as tam_analysis.py (teacher-force the model's own generated
prediction, take the last decoder layer's cross-attention over the source
span, same feature-index position mapping) but also backprops the
teacher-forcing loss to that attention tensor:

    GradCAM(position) = ReLU(attn * d(loss)/d(attn))

See tam_analysis.py's module docstring for:
  - the exact fairseq API path used to obtain last-layer, head-averaged
    cross-attention from an ordinary forward pass (extra["attn"][0] via
    TransformerDecoderBase.extract_features_scriptable's unconditional
    need_attn/need_head_weights=True call at the last layer) -- reused
    verbatim here, and
  - why the position mapping is the identity (source index == I3D feature
    row index, no subsampling in Sign2TextTransformerEncoder) and why the
    feature-index -> real-video-frame/time conversion is NOT attempted
    (upstream I3D window/stride not present in this repo).

*** DRAFT / UNTESTED: no trained checkpoint exists yet. Only checked with
`python -m py_compile`. ***

Loss used for backprop
-----------------------
SpaMo backprops `out.loss` from HF's T5ForConditionalGeneration, which
computes label-smoothed-free per-token cross-entropy internally. This repo
trains sign2text_transformer with fairseq's `label_smoothed_cross_entropy`
criterion (label_smoothing: 0.0 in examples/sign_language/config/
wicv_cvpr23/i3d_best/baseline_6_3.yaml, i.e. effectively plain NLL for the
checkpoints trained with that config) via fairseq's Criterion/Trainer
machinery, which is awkward to invoke standalone outside a full
Task+Trainer setup. Rather than reconstruct fairseq's criterion registry
here, this script computes an equivalent plain per-sentence NLL loss
directly with `torch.nn.functional.nll_loss` over `log_softmax(logits)`,
ignoring pad positions -- mathematically identical to
label_smoothed_cross_entropy with label_smoothing=0.0 summed over tokens.
TODO/ASSUMPTION: if a checkpoint was actually trained with label_smoothing
> 0, this loss's *gradient direction* w.r.t. attention should still be
essentially the same (label smoothing only reweights the target
distribution slightly), but this has not been verified numerically against
fairseq's actual criterion.

Outputs (under analysis/predictions/):
    gradcam_stats.csv    -- per-sentence: sentence_id, n_tokens, mean_entropy, mean_peak_relpos
    gradcam_summary.csv  -- corpus-level stats + Spearman r vs. this same sentence's
                             TAM peakiness (if tam_stats.csv exists)
    gradcam_full.json    -- all sentences' full per-token GradCAM rows

Usage (pick GPU 4/5/6/7; always nohup -- see project memory):
    conda activate <sign2text env>
    cd external/slt_how2sign_wicv2023
    CUDA_VISIBLE_DEVICES=4 python analysis/gradcam_analysis.py \
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
import torch.nn.functional as F
from scipy import stats

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fairseq import checkpoint_utils  # noqa: E402
from tam_analysis import (  # noqa: E402
    DEFAULT_TEST_SPLIT, PREDICTIONS_DIR, PREDICTIONS_CSV,
    load_model_and_task, encode_target, token_str,
)

TAM_STATS_CSV = os.path.join(PREDICTIONS_DIR, "tam_stats.csv")


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
    underlying = dataset.dataset  # AddTargetDataset -> SignFeatsDataset
    id_to_idx = {vid_id: i for i, vid_id in enumerate(underlying.ids)}

    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"{PREDICTIONS_CSV} not found -- run run_inference.py first to produce it."
        )
    preds_df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Running GradCAM extraction over {len(preds_df)} test sentences...")

    pad_idx = task.target_dictionary.pad()
    # See fairseq/modules/multihead_attention.py's `_capture_attn_grad` hook
    # (local patch, additive-only -- no change to model behavior): the
    # `extra["attn"]` fairseq normally returns is a dead-end copy that never
    # receives a gradient from loss.backward(), so we instead grab the
    # actual attn_probs tensor consumed by the attention's bmm via this hook.
    last_encoder_attn = model.decoder.layers[-1].encoder_attn
    last_encoder_attn._capture_attn_grad = True
    num_heads = last_encoder_attn.num_heads
    rows = []
    full_examples = []
    for n_done, rec in enumerate(preds_df.itertuples(), 1):
        sid = rec.SENTENCE_ID
        if sid not in id_to_idx:
            print(f"  WARNING: {sid} not found in dataset, skipping")
            continue
        idx = id_to_idx[sid]
        sample = dataset[idx]
        src = sample["source"]
        if src.shape[0] == 0:
            continue
        real_src_len = src.shape[0]

        src_tokens = src.unsqueeze(0).float().to(device)
        encoder_padding_mask = torch.zeros(1, real_src_len, dtype=torch.bool, device=device)

        pred_text = str(rec.prediction)
        prev_output_tokens, target = encode_target(task, pred_text, device)

        model.zero_grad(set_to_none=True)
        last_encoder_attn._captured_attn_probs = None
        logits, extra = model(
            src_tokens=src_tokens,
            encoder_padding_mask=encoder_padding_mask,
            prev_output_tokens=prev_output_tokens,
        )
        attn_probs = last_encoder_attn._captured_attn_probs  # [bsz*num_heads, tgt_len, src_len] (bsz=1 here)
        if attn_probs is None:
            print(f"  WARNING: {sid} produced no attention (capture hook empty), skipping")
            continue

        lprobs = F.log_softmax(logits.float(), dim=-1)  # [1, tgt_len, vocab]
        loss = F.nll_loss(
            lprobs.view(-1, lprobs.size(-1)), target.view(-1),
            ignore_index=pad_idx, reduction="sum",
        )
        loss.backward()
        grad = attn_probs.grad
        if grad is None:
            print(f"  WARNING: {sid} got no gradient on cross-attention, skipping")
            continue

        # attn_probs is [num_heads, tgt_len, src_len] (bsz=1); average over
        # heads to match tam_analysis.py's / fairseq's own head-averaged
        # convention before computing ReLU(attn * grad).
        tgt_len = attn_probs.size(1)
        cam_per_head = torch.relu(attn_probs.detach() * grad).view(num_heads, tgt_len, -1)
        cam = cam_per_head.mean(dim=0).float().cpu().numpy()  # [tgt_len, src_len]
        cam = cam[:, :real_src_len]
        row_sums = cam.sum(axis=1, keepdims=True)
        cam = np.divide(cam, row_sums, out=np.zeros_like(cam), where=row_sums > 1e-12)

        target_ids = target[0].tolist()

        entropies, peak_relpos, token_rows = [], [], []
        for t_idx, tok_id in enumerate(target_ids):
            if tok_id == pad_idx or t_idx >= cam.shape[0]:
                continue
            row = cam[t_idx]
            if row.sum() <= 1e-12:
                continue
            max_entropy = np.log(real_src_len) if real_src_len > 1 else 1.0
            ent = -np.sum(row * np.log(row.clip(min=1e-12)))
            entropies.append(ent / max_entropy)
            peak_j = int(np.argmax(row))
            peak_relpos.append(peak_j / max(real_src_len - 1, 1))
            token_rows.append(dict(
                token=token_str(task, tok_id),
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
    stats_path = os.path.join(PREDICTIONS_DIR, "gradcam_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"wrote {stats_path} ({len(stats_df)} sentences)")

    full_path = os.path.join(PREDICTIONS_DIR, "gradcam_full.json")
    with open(full_path, "w") as f:
        json.dump(full_examples, f)
    print(f"wrote {full_path} ({len(full_examples)} sentences, full per-token GradCAM)")

    r_tam, p_tam = float("nan"), float("nan")
    n_matched_tam = 0
    if os.path.exists(TAM_STATS_CSV) and len(stats_df):
        tam_df = pd.read_csv(TAM_STATS_CSV)[["sentence_id", "mean_entropy"]].rename(
            columns={"mean_entropy": "tam_entropy"}
        )
        both = stats_df.merge(tam_df, on="sentence_id")
        n_matched_tam = len(both)
        if n_matched_tam > 5:
            r_tam, p_tam = stats.spearmanr(1 - both["mean_entropy"], 1 - both["tam_entropy"])

    summary = dict(
        n_sentences=len(stats_df),
        mean_entropy=stats_df["mean_entropy"].mean() if len(stats_df) else float("nan"),
        median_entropy=stats_df["mean_entropy"].median() if len(stats_df) else float("nan"),
        mean_peak_relpos=stats_df["mean_peak_relpos"].mean() if len(stats_df) else float("nan"),
        spearman_r_vs_tam_peakiness=r_tam,
        spearman_p_vs_tam=p_tam,
        n_matched_vs_tam=n_matched_tam,
    )
    summary_path = os.path.join(PREDICTIONS_DIR, "gradcam_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print("\n=== GradCAM summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {summary_path}")
    if not os.path.exists(TAM_STATS_CSV):
        print("NOTE: tam_stats.csv not found -- run tam_analysis.py first for the TAM-vs-GradCAM cross-check.")
    print(
        "NOTE: unlike SpaMo, there is no region-masking-vs-peakiness comparison here "
        "(no masked-feature-extraction analysis exists yet for this model)."
    )


if __name__ == "__main__":
    main()
