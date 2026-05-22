"""
Collaborative Recommender
Uses Truncated SVD (matrix factorization) on the user-item interaction
matrix to discover latent factors and predict ratings.

Improvements:
- Implicit feedback support (views, purchases → confidence weights)
- Adaptive n_factors for sparse matrices
- User-based personalized recommendations
- [NEW] NeuMF (Neural Matrix Factorization) — two-tower ANN replacing SVD
         Enable via USE_NEUMF=true in .env
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix

# ── NeuMF imports (only needed when USE_NEUMF=true) ───────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

USE_NEUMF = os.getenv("USE_NEUMF", "false").lower() == "true"


# ══════════════════════════════════════════════════════════════════════════
#  1. NeuMF architecture
# ══════════════════════════════════════════════════════════════════════════

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization — He et al. 2017
    https://arxiv.org/abs/1708.05031

    HOW IT TRACKS USER BEHAVIOUR
    ─────────────────────────────
    Every user gets two learned embedding vectors (one for GMF, one for MLP).
    Each time the model sees a positive interaction (user bought/clicked item),
    backprop nudges those vectors so the user embedding moves closer to the
    items they engaged with in the latent space.

    At inference: score all items for a user → ranked personalised list
    that reflects their full interaction history.

    TWO TOWERS
    ──────────
    GMF branch  — element-wise product of user & item embeddings.
                  Learns the same linear signal as SVD but end-to-end.

    MLP branch  — concat(user_emb, item_emb) → Dense+ReLU layers.
                  Learns non-linear patterns SVD can NEVER capture,
                  e.g. "users who buy hiking boots + rain jacket
                        probably also want trekking poles".

    Both outputs are concatenated → single Linear → Sigmoid → score ∈ (0,1)
    """
    def __init__(self, n_users, n_items,
                 emb_dim=64,
                 mlp_layers=(256, 128, 64),
                 dropout=0.2):
        super().__init__()

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(n_items, emb_dim)

        # MLP embeddings  (kept separate — sharing hurts quality in practice)
        self.mlp_user_emb = nn.Embedding(n_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, emb_dim)

        # MLP tower
        layers, in_dim = [], emb_dim * 2
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Fusion layer
        self.output_layer = nn.Linear(emb_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        # GMF branch
        gmf_out = self.gmf_user_emb(user_ids) * self.gmf_item_emb(item_ids)

        # MLP branch
        mlp_in  = torch.cat([self.mlp_user_emb(user_ids),
                              self.mlp_item_emb(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Fuse + score
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.sigmoid(self.output_layer(fused)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════
#  2. Implicit feedback dataset  (positive + sampled negatives)
# ══════════════════════════════════════════════════════════════════════════

class ImplicitFeedbackDataset(Dataset):
    """
    For each positive (user, item) interaction → label = 1
    Sample neg_ratio random items the user has NOT touched → label = 0

    This is the standard NCF training strategy for implicit feedback:
    model learns "liked" vs "never interacted with",
    NOT explicit star ratings.
    """
    def __init__(self, interactions, n_items, neg_ratio=4):
        self.data = []
        item_set  = set(range(n_items))

        user_positives = {}
        for u, i in interactions:
            user_positives.setdefault(u, set()).add(i)

        for u, pos_items in user_positives.items():
            negatives = list(item_set - pos_items)
            for i in pos_items:
                self.data.append((u, i, 1.0))
                negs = np.random.choice(negatives,
                                        size=min(neg_ratio, len(negatives)),
                                        replace=False)
                for n in negs:
                    self.data.append((u, int(n), 0.0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, label = self.data[idx]
        return (torch.tensor(u,     dtype=torch.long),
                torch.tensor(i,     dtype=torch.long),
                torch.tensor(label, dtype=torch.float32))


# ══════════════════════════════════════════════════════════════════════════
#  3. NeuMFTrainer  — same public interface as CollaborativeRecommender
#     so hybrid_model.py needs ZERO changes
# ══════════════════════════════════════════════════════════════════════════

class NeuMFTrainer:
    def __init__(
        self,
        interaction_df,
        emb_dim=64,
        mlp_layers=(256, 128, 64),
        dropout=0.2,
        epochs=20,
        batch_size=256,
        lr=1e-3,
        neg_ratio=4,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch not installed. Run: pip install torch\n"
                "Or set USE_NEUMF=false to fall back to TruncatedSVD."
            )

        self.df = interaction_df.copy()

        # Build index mappings  (identical to CollaborativeRecommender)
        self.users  = self.df['user_id'].astype('category')
        self.titles = self.df['title'].astype('category')

        self._user_to_idx  = {u: i for i, u in enumerate(self.users.cat.categories)}
        self._title_to_idx = {t: i for i, t in enumerate(self.titles.cat.categories)}
        self.title_list    = list(self.titles.cat.categories)

        n_users = len(self._user_to_idx)
        n_items = len(self._title_to_idx)

        interactions = list(zip(self.users.cat.codes.values,
                                self.titles.cat.codes.values))

        dataset    = ImplicitFeedbackDataset(interactions, n_items, neg_ratio)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = NeuMF(n_users, n_items, emb_dim, mlp_layers, dropout).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        print(f"[NeuMF] device={self.device} | "
              f"users={n_users} | items={n_items} | samples={len(dataset)}")
        self._train(epochs)

    # ── Training loop ──────────────────────────────────────────────────

    def _train(self, epochs):
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for user_ids, item_ids, labels in self.loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels   = labels.to(self.device)

                self.optimiser.zero_grad()
                loss = self.criterion(self.model(user_ids, item_ids), labels)
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}/{epochs}  "
                      f"loss={total_loss / len(self.loader):.4f}")

    # ── Score all items for one user (used by both recommend methods) ──

    @torch.no_grad()
    def _score_all_items(self, user_idx):
        self.model.eval()
        n = len(self.title_list)
        u = torch.tensor([user_idx] * n, dtype=torch.long).to(self.device)
        i = torch.arange(n,              dtype=torch.long).to(self.device)
        return self.model(u, i).cpu().numpy()

    # ── Public interface (mirrors CollaborativeRecommender exactly) ────

    def recommend(self, title, top_n=10):
        """
        Item-item recommendations via user behaviour patterns.
        Finds all users who interacted with `title`, averages their
        predicted score vectors across all items, returns top_n.
        """
        if title not in self._title_to_idx:
            return []

        item_idx       = self._title_to_idx[title]
        users_who_liked = self.df[self.df['title'] == title]['user_id'].unique()

        if len(users_who_liked) == 0:
            return []

        score_matrix = np.stack([
            self._score_all_items(self._user_to_idx[u])
            for u in users_who_liked
            if u in self._user_to_idx
        ])
        avg_scores           = score_matrix.mean(axis=0)
        avg_scores[item_idx] = -1.0          # exclude the query item

        top_indices = np.argsort(avg_scores)[::-1][:top_n]
        return [
            {'title': self.title_list[i], 'collab_score': float(avg_scores[i])}
            for i in top_indices
        ]

    def predict_for_user(self, user_id, top_n=10):
        """Personalised recommendations for a specific user."""
        if user_id not in self._user_to_idx:
            return []

        scores     = self._score_all_items(self._user_to_idx[user_id])
        seen_items = set(self.df[self.df['user_id'] == user_id]['title'].tolist())

        scored = [
            (self.title_list[i], float(scores[i]))
            for i in range(len(self.title_list))
            if self.title_list[i] not in seen_items
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{'title': t, 'predicted_score': s} for t, s in scored[:top_n]]

    def predict_rating(self, user_id, title):
        """Predict interaction probability for a user-item pair → float ∈ (0,1)."""
        if user_id not in self._user_to_idx or title not in self._title_to_idx:
            return None
        self.model.eval()
        with torch.no_grad():
            u = torch.tensor([self._user_to_idx[user_id]], dtype=torch.long).to(self.device)
            i = torch.tensor([self._title_to_idx[title]],  dtype=torch.long).to(self.device)
            return float(self.model(u, i).item())


# ══════════════════════════════════════════════════════════════════════════
#  4. Original SVD recommender  (unchanged — kept as fallback)
# ══════════════════════════════════════════════════════════════════════════

class CollaborativeRecommender:
    def __init__(self, interaction_df, n_factors=50, use_implicit=True):
        """
        interaction_df: DataFrame with columns 'user_id', 'title', 'rating'.
                        Optionally 'views' and 'purchases' for implicit feedback.
        n_factors: number of latent factors for SVD decomposition.
        use_implicit: blend in implicit feedback signals if available.
        """
        self.df = interaction_df.copy()

        self.users  = self.df['user_id'].astype('category')
        self.titles = self.df['title'].astype('category')

        self._user_to_idx  = {u: i for i, u in enumerate(self.users.cat.categories)}
        self._title_to_idx = {t: i for i, t in enumerate(self.titles.cat.categories)}
        self.title_list    = list(self.titles.cat.categories)

        row  = self.users.cat.codes.values
        col  = self.titles.cat.codes.values
        data = self.df['rating'].values.astype(float)

        if use_implicit:
            alpha_implicit = 0.5
            if 'purchases' in self.df.columns:
                data = data + alpha_implicit * self.df['purchases'].fillna(0).values
            if 'views' in self.df.columns:
                data = data + (alpha_implicit * 0.5) * self.df['views'].fillna(0).values

        n_users = len(self._user_to_idx)
        n_items = len(self._title_to_idx)
        self.user_item_sparse = coo_matrix(
            (data, (row, col)), shape=(n_users, n_items)
        ).tocsr()

        min_dim = min(self.user_item_sparse.shape)
        density = (self.user_item_sparse.nnz / (n_users * n_items)
                   if (n_users * n_items) > 0 else 0)

        if density < 0.001:
            n_components = min(20, min_dim - 1)
        elif density < 0.01:
            n_components = min(30, min_dim - 1)
        else:
            n_components = min(n_factors, min_dim - 1)
        n_components = max(1, n_components)

        self.svd          = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_item_sparse)
        self.item_factors = self.svd.components_

    def recommend(self, title, top_n=10):
        if title not in self._title_to_idx:
            return []
        idx       = self._title_to_idx[title]
        query_vec = self.item_factors[:, idx].reshape(1, -1)
        scores    = cosine_similarity(query_vec, self.item_factors.T).flatten()
        sim_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results, seen = [], set()
        for i, score in sim_scores:
            t = self.title_list[i]
            if t == title or t in seen:
                continue
            seen.add(t)
            results.append({'title': t, 'collab_score': float(score)})
            if len(results) >= top_n:
                break
        return results

    def predict_for_user(self, user_id, top_n=10):
        if user_id not in self._user_to_idx:
            return []
        u_idx      = self._user_to_idx[user_id]
        scores     = np.dot(self.user_factors[u_idx], self.item_factors)
        seen_items = set(self.df[self.df['user_id'] == user_id]['title'].tolist())
        scored = [(self.title_list[i], float(s))
                  for i, s in enumerate(scores)
                  if self.title_list[i] not in seen_items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{'title': t, 'predicted_score': s} for t, s in scored[:top_n]]

    def predict_rating(self, user_id, title):
        if user_id not in self._user_to_idx or title not in self._title_to_idx:
            return None
        u_idx = self._user_to_idx[user_id]
        i_idx = self._title_to_idx[title]
        return float(np.dot(self.user_factors[u_idx], self.item_factors[:, i_idx]))


# ══════════════════════════════════════════════════════════════════════════
#  5. Factory function  — the only thing hybrid_model.py needs to change
# ══════════════════════════════════════════════════════════════════════════

def get_collaborative_recommender(interaction_df, **kwargs):
    """
    Call this instead of CollaborativeRecommender() directly.

    Returns NeuMFTrainer  if USE_NEUMF=true  in .env
    Returns CollaborativeRecommender (SVD)  otherwise  (default, safe)

    hybrid_model.py change needed:
        # Before
        collab = CollaborativeRecommender(interaction_df)
        # After
        from collaborative_model import get_collaborative_recommender
        collab = get_collaborative_recommender(interaction_df)
    """
    if USE_NEUMF:
        if not TORCH_AVAILABLE:
            print("[NeuMF] WARNING: torch not found — falling back to SVD.")
            return CollaborativeRecommender(interaction_df, **kwargs)
        print("[NeuMF] USE_NEUMF=true — Neural Matrix Factorization active")
        return NeuMFTrainer(interaction_df)
    print("[SVD]   USE_NEUMF=false — TruncatedSVD active (default)")
    return CollaborativeRecommender(interaction_df, **kwargs)