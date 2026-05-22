"""
Content-Based Recommender
Uses TF-IDF vectorization on item metadata (title + description + category)
and cosine similarity to find similar items.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, item_df):
        """
        item_df: DataFrame with at least 'title' and 'combined' columns.
        'combined' = title + description + category (created by data_adapter).
        """
        self.df = item_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
        )
        if "combined" not in self.df.columns:
            self.df["combined"] = (
                self.df.get("title", "").astype(str) + " " +
                self.df.get("author", "").astype(str) + " " +
                self.df.get("category", "").astype(str)
            )
        self.matrix = self.vectorizer.fit_transform(self.df['combined'].fillna(''))
        # Do not compute full similarity matrix here to avoid OOM
        self._title_to_idx = {
            t.lower(): i for i, t in enumerate(self.df['title'].astype(str))
        }

    def recommend(self, title, top_n=10):
        """
        Get content-based recommendations for a given item title.
        Returns list of dicts: [{ 'title', 'content_score' }, ...]
        """
        if title.lower() not in self._title_to_idx:
            return []

        title_key = title.lower()

        if title_key not in self._title_to_idx:
            return []

        idx = self._title_to_idx[title_key]

    
        query_vec = self.matrix[idx]
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        
        scores[idx] = float("-inf")

        top_indices = np.argsort(-scores)

        results = []
        seen = set() 
        for i in top_indices+1:
            t = str(self.df.iloc[i]["title"]).strip()
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            
            results.append({
                "title": t,
                "content_score": float(scores[i])
            })
            
            if len(results) >= top_n:
                break

        return results

    def explain_similarity(self, source_title, candidate_title, top_n=5):
        """Return top TF-IDF terms shared by the source and candidate item."""
        if source_title not in self._title_to_idx or candidate_title not in self._title_to_idx:
            return []

        source_idx = self._title_to_idx[source_title]
        candidate_idx = self._title_to_idx[candidate_title]
        contributions = self.matrix[source_idx].multiply(self.matrix[candidate_idx]).toarray().ravel()
        if not np.any(contributions):
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = contributions.argsort()[::-1]
        terms = []
        for idx in top_indices:
            score = float(contributions[idx])
            if score <= 0:
                break
            terms.append({
                'term': str(feature_names[idx]),
                'score': round(score, 4),
            })
            if len(terms) >= top_n:
                break
        return terms

    def search(self, query, top_n=20):
        """
        Search items by query text using TF-IDF similarity.
        Returns list of matching item titles with scores.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argpartition(-scores, top_n)[:top_n]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        seen = set()
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            t = self.df.iloc[idx]['title']
            if t in seen:
                continue
            seen.add(t)
            
            tp = self.df.at[idx, 'top_reviews'] if 'top_reviews' in self.df.columns else []
            top_reviews = tp if isinstance(tp, list) else []

            results.append({
                'title': t,
                'score': float(scores[idx]),
                'item_id': str(self.df.iloc[idx].get('item_id', idx)),
                'category': self.df.iloc[idx].get('category', ''),
                'description': str(self.df.iloc[idx].get('description', ''))[:200],
                'top_reviews': top_reviews,
            })
        return results
