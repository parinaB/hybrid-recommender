"""
evaluation.py — offline evaluation of the hybrid recommender.

Fixes vs. previous version:
  - seed_item is now resolved to a real title via item_df before being
    passed to recommend(). Ratings rows carry ISBNs/IDs as their title;
    without resolution every recommend() call silently fails (item not
    found in content model) and the results table is empty.
  - relevant set is also ID→title resolved so P/R/NDCG are computed in
    the same title space as the recommendations.
  - Added per-config skip count column so silent failures surface.
  - Debug block prints resolved seed title so the fix is easy to verify.
"""

import os
import sys
import random
import numpy as np
from math import log2

sys.path.insert(0, os.path.dirname(__file__))

from dataset_manager import DatasetManager
from nlp_engine import batch_analyze, aggregate_sentiment_by_item
from content_model import ContentRecommender
from collaborative_model import get_collaborative_recommender
from hybrid_model import HybridRecommender
from src.data.dataset_manager import DatasetManager
from src.model.nlp_engine import batch_analyze, aggregate_sentiment_by_item
from src.model.content_model import ContentRecommender
from src.model.collaborative_model import CollaborativeRecommender
from src.model.hybrid_model import HybridRecommender


# ─────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────

def precision_at_k(rec, rel, k):
    rec = rec[:k]
    return len(set(rec) & set(rel)) / k if k else 0.0


def recall_at_k(rec, rel, k):
    rec = rec[:k]
    return len(set(rec) & set(rel)) / len(rel) if rel else 0.0


def ndcg_at_k(rec, rel, k):
    dcg  = sum(1 / log2(i + 2) for i, x in enumerate(rec[:k]) if x in rel)
    idcg = sum(1 / log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


# ─────────────────────────────────────────────
#  Title resolution helpers
# ─────────────────────────────────────────────
def average_precision_at_k(recommended, relevant, k):
    """Average Precision @ K for a single query."""
    rec_k = recommended[:k]
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    
    return sum_precisions / min(len(relevant), k) if relevant else 0.0


def evaluate():
    """Run the full evaluation pipeline."""
    # 1. Load data
    dm = DatasetManager()
    data_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    
    # Try to load all user-provided datasets first
    datasets_to_load = ['books.csv', 'booksdata.csv', 'ratings.csv']
    loaded_any = False
    
    for filename in datasets_to_load:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading dataset: {filename}...")
            dm.load_csv(filepath)
            loaded_any = True
            
    # Fallback to sample data if no user datasets found
    if not loaded_any:
        sample_file = os.path.join(data_dir, 'sample_products.csv')
        if not os.path.exists(sample_file):
            print("ERROR: datasets not found. Run: python scripts/generate_sample_data.py")
            return
        print("Loading sample_products.csv...")
        dm.load_csv(sample_file)

def build_id_to_title_map(interaction_df):
    """
    Return {item_id_str -> real_title} built from rows where the title
    is not the same as the item_id (i.e. a real book title was detected).
    """
    if 'item_id' not in interaction_df.columns:
        return {}

    real = interaction_df[
        interaction_df['title'].astype(str) != interaction_df['item_id'].astype(str)
    ][['item_id', 'title']].drop_duplicates('item_id')

    return dict(zip(real['item_id'].astype(str), real['title'].astype(str)))


def resolve_title(raw, id_to_title, valid_titles):
    """
    Map a raw value (real title or ISBN/ID) to a title present in
    item_df.  Returns None if no match can be found.
    """
    raw = str(raw)
    if raw in valid_titles:
        return raw
    resolved = id_to_title.get(raw)
    if resolved and resolved in valid_titles:
        return resolved
    return None


# ─────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────

def evaluate(
    max_nlp_rows: int = 2000,
    max_test_users: int = 500,
    min_interactions: int = 5,
    relevance_threshold: float = 3.0,
    K: int = 10,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    # ── 1. Load datasets ──────────────────────────────────────────────
    dm = DatasetManager()
    data_dir = os.path.join(os.path.dirname(__file__), "datasets")

    loaded = False
    for fname in ["books.csv", "booksdata.csv", "ratings.csv"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            print(f"Loading {fname} …")
            dm.load_csv(path)
            loaded = True

    if not loaded:
        print("❌  No dataset CSV found in", data_dir)
        return

    interaction_df, item_df = dm.merge_all()
    print(f"Total rows      : {len(interaction_df):,}")
    print(f"Unique users    : {interaction_df['user_id'].nunique():,}")
    print(f"Unique items    : {interaction_df['title'].nunique():,}")

    if interaction_df['user_id'].nunique() < 2:
        print("❌  Not enough users — check dataset merge.")
        return

    # ── 2. NLP sentiment enrichment (sample only, for speed) ─────────
    print(f"\nRunning NLP on {min(max_nlp_rows, len(interaction_df)):,} rows …")
    nlp_sample = batch_analyze(interaction_df.head(max_nlp_rows), "review_text")
    sentiment  = aggregate_sentiment_by_item(nlp_sample, "title")
    item_df    = item_df.merge(sentiment, on="title", how="left")
    item_df["avg_sentiment"] = item_df["avg_sentiment"].fillna(0.0)

    # ── 3. Build ID → real-title lookup ──────────────────────────────
    # item_df is the ground-truth title space for the content model.
    # interaction_df may have bare ISBNs/IDs as titles (from ratings.csv).
    valid_titles = set(item_df["title"].astype(str).tolist())
    id_to_title  = build_id_to_title_map(interaction_df)

    print(f"Title map size  : {len(id_to_title):,}  id→title entries")
    print(f"Valid titles    : {len(valid_titles):,}")

    # ── 4. Build test set ─────────────────────────────────────────────
    print("\nBuilding test set …")
    test_pairs = []   # (user_id, resolved_seed_title, [resolved_relevant_titles])

    for user_id, group in interaction_df.groupby("user_id"):
      if len(group) < min_interactions:
        continue

      group_sorted = group.sort_values("rating", ascending=False)

      seed_raw = group_sorted.iloc[0]["title"]
      seed_title = resolve_title(seed_raw, id_to_title, valid_titles)

    # fallback: if not resolved, try direct match
      if seed_title is None and seed_raw in valid_titles:
        seed_title = seed_raw

      if seed_title is None:
        continue

    # relevant items
      high_rated_raw = group[group["rating"] >= relevance_threshold]["title"].tolist()

      resolved_rel = [
        resolve_title(x, id_to_title, valid_titles)
        for x in high_rated_raw
      ]
      resolved_rel = [x for x in resolved_rel if x is not None]

    # fallback if empty
      if not resolved_rel:
        median_r = group["rating"].median()
        fallback_raw = group[group["rating"] >= median_r]["title"].tolist()

        resolved_rel = [
            resolve_title(x, id_to_title, valid_titles)
            for x in fallback_raw
        ]
        resolved_rel = [x for x in resolved_rel if x is not None]

      if not resolved_rel:
        continue

      test_pairs.append((user_id, seed_title, resolved_rel))

    print(f"Eligible test users : {len(test_pairs):,}")

    if len(test_pairs) < 2:
        print(
            "❌  Not enough evaluation data — "
            "try lowering min_interactions or relevance_threshold."
        )
        return

    if len(test_pairs) > max_test_users:
        test_pairs = random.sample(test_pairs, max_test_users)
        print(f"Sampled to          : {len(test_pairs):,} users")

    # ── 5. Initialise models ──────────────────────────────────────────
    print("\nInitialising models …")
    content_model = ContentRecommender(item_df)
    collab_model  = get_collaborative_recommender(interaction_df)

    # ── 6. Debug: smoke-test one recommend() call ─────────────────────
    print("\n[DEBUG] Testing a single recommend() call …")
    test_uid, test_item, test_rel = test_pairs[0]
    print(f"  user_id      : {test_uid}")
    print(f"  seed_item    : {test_item!r}")
    print(f"  relevant[:3] : {test_rel[:3]}")

    try:
        hybrid_debug = HybridRecommender(content_model, collab_model, item_df, 0.5, 0.5, 0.0)
        recs_debug   = hybrid_debug.recommend(test_item, top_n=10)
        print(f"  recs returned : {len(recs_debug)}")
        print(f"  sample recs   : {[x['title'] for x in recs_debug[:3]]}")
    except Exception:
        import traceback
        print("  ❌  recommend() raised an exception:")
        traceback.print_exc()
        print("  ⚠️   Continuing with full evaluation.\n")

    # ── 7. Evaluate configurations ────────────────────────────────────
    configs = [
        ("Alpha=0.3", 0.3, 0.7, 0.0),
        ("Alpha=0.5", 0.5, 0.5, 0.0),
        ("Alpha=0.7", 0.7, 0.3, 0.0),
    ]

    header = f"{'Config':<12} {'P@'+str(K):<10} {'R@'+str(K):<10} {'NDCG@'+str(K):<10} {'Users':<8} Skipped"
    print(f"\n{header}")
    print("-" * len(header))

    results = {}
    for name, a, b, g in configs:
        hybrid = HybridRecommender(content_model, collab_model, item_df, a, b, g)

        p_list, r_list, n_list = [], [], []
        skipped = 0

        for user_id, seed_item, rel in test_pairs:
            try:
                recs       = hybrid.recommend(seed_item, top_n=K)
                rec_titles = [x["title"] for x in recs]
            except Exception:
                skipped += 1
                continue

            p_list.append(precision_at_k(rec_titles, rel, K))
            r_list.append(recall_at_k(rec_titles, rel, K))
            n_list.append(ndcg_at_k(rec_titles, rel, K))

        evaluated = len(p_list)
        if evaluated == 0:
            print(f"{name:<12} — no successful recommendations (all {skipped} skipped)")
            continue

        p_mean = np.mean(p_list)
        r_mean = np.mean(r_list)
        n_mean = np.mean(n_list)

        print(
            f"{name:<12} {p_mean:<10.4f} {r_mean:<10.4f} {n_mean:<10.4f} "
            f"{evaluated:<8} {skipped}"
        )
        results[name] = {"P@K": p_mean, "R@K": r_mean, "NDCG@K": n_mean, "n": evaluated}

    # ── 8. Summary ────────────────────────────────────────────────────
    if results:
        best = max(results, key=lambda x: results[x]["NDCG@K"])
        print(
            f"\n✅  Best config by NDCG@{K}: {best}  "
            f"(NDCG = {results[best]['NDCG@K']:.4f})"
        )
    else:
        print("\n⚠️   No results produced — review model outputs.")


if __name__ == "__main__":
    evaluate()
