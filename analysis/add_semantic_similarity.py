"""
add_semantic_similarity.py

Fills in the `semantic_similarity` column (sentence-transformers cosine
similarity between prediction and GT, "all-MiniLM-L6-v2") across this
model's analysis/predictions/predictions_*.csv files -- same convention as
external/SpaMo/analysis/add_semantic_similarity.py and
external/vtamo/analysis/add_semantic_similarity.py, copied verbatim (this
model has no torch/transformers version conflict of its own with
sentence-transformers, but keeping it a separate script matches the
established per-model pattern and still needs the `mmslt` conda env, which
is the one with a working sentence-transformers install in this workspace).

Usage:
    conda activate mmslt
    python external/slt_how2sign_wicv2023/analysis/add_semantic_similarity.py
"""
import csv
import glob
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(REPO_DIR, "analysis", "predictions")


def get_semantic_similarity_fn():
    from sentence_transformers import SentenceTransformer, util
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def fn(preds, refs):
        emb_p = st_model.encode(preds, convert_to_tensor=True, show_progress_bar=False)
        emb_r = st_model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(emb_p, emb_r).diagonal()
        return [float(s) for s in sims]

    return fn


def process_file(path, sim_fn):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{path}: empty, skipping")
        return

    if "semantic_similarity" in rows[0] and all(r.get("semantic_similarity") for r in rows):
        print(f"{path}: already has semantic_similarity, skipping")
        return

    preds = [r["prediction"] for r in rows]
    refs = [r["GT"] for r in rows]
    sims = sim_fn(preds, refs)

    fieldnames = list(rows[0].keys())
    if "semantic_similarity" not in fieldnames:
        fieldnames.append("semantic_similarity")
    for r, s in zip(rows, sims):
        r["semantic_similarity"] = f"{s:.4f}"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: added semantic_similarity for {len(rows)} rows "
          f"(mean={sum(sims) / len(sims):.4f})")


def main():
    paths = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "predictions_*.csv")))
    print(f"Found {len(paths)} predictions files under {PREDICTIONS_DIR}")
    sim_fn = get_semantic_similarity_fn()
    for p in paths:
        process_file(p, sim_fn)


if __name__ == "__main__":
    main()
