# recommender_models.py
# Unified module for all recommenders + persistence + registry

import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Persistence Layer (uses pickle by default)
# ---------------------------------------------------------------------------
class ModelIO:
    @staticmethod
    def save(model, path: str):
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


class PersistentRecommenderMixin:
    def save(self, path: str):
        ModelIO.save(self, path)

    @classmethod
    def load(cls, path: str):
        return ModelIO.load(path)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, model: Any):
        cls._registry[name] = model

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._registry.get(name)

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())


# ---------------------------------------------------------------------------
# Base Recommender Interface
# ---------------------------------------------------------------------------
class BaseRecommender:
    def fit(self, ratings: pd.DataFrame):
        raise NotImplementedError

    def predict(self, user: int, item: int) -> float:
        raise NotImplementedError

    def recommend(self, user: int, n: int = 10):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baseline Bias Model
# ---------------------------------------------------------------------------
class BaselineBias(PersistentRecommenderMixin, BaseRecommender):
    def fit(self, ratings: pd.DataFrame):
        self.global_mean = ratings['rating'].mean()
        self.user_bias = ratings.groupby('user_id')['rating'].mean() - self.global_mean
        self.item_bias = ratings.groupby('item_id')['rating'].mean() - self.global_mean
        return self

    def predict(self, user, item):
        ub = self.user_bias.get(user, 0.0)
        ib = self.item_bias.get(item, 0.0)
        return float(self.global_mean + ub + ib)

    def recommend(self, user, n=10):
        items = self.item_bias.index
        preds = [(item, self.predict(user, item)) for item in items]
        return sorted(preds, key=lambda x: -x[1])[:n]


# ---------------------------------------------------------------------------
# SVD Recommender (naive implementation)
# ---------------------------------------------------------------------------
class SVDRecommender(PersistentRecommenderMixin, BaseRecommender):
    def __init__(self, n_factors=20, n_iters=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.lr = lr
        self.reg = reg

    def fit(self, ratings: pd.DataFrame):
        users = ratings['user_id'].unique()
        items = ratings['item_id'].unique()

        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {m: i for i, m in enumerate(items)}
        self.user_rev = {i: u for u, i in self.user_map.items()}
        self.item_rev = {i: m for m, i in self.item_map.items()}

        n_users = len(users)
        n_items = len(items)
        k = self.n_factors

        self.U = np.random.normal(scale=0.1, size=(n_users, k))
        self.V = np.random.normal(scale=0.1, size=(n_items, k))

        for _ in range(self.n_iters):
            for _, row in ratings.iterrows():
                u = self.user_map[row['user_id']]
                m = self.item_map[row['item_id']]
                r = row['rating']

                pred = np.dot(self.U[u], self.V[m])
                err = r - pred

                self.U[u] += self.lr * (err * self.V[m] - self.reg * self.U[u])
                self.V[m] += self.lr * (err * self.U[u] - self.reg * self.V[m])

        return self

    def predict(self, user, item):
        if user not in self.user_map or item not in self.item_map:
            return np.nan
        return float(np.dot(self.U[self.user_map[user]], self.V[self.item_map[item]]))

    def recommend(self, user, n=10):
        if user not in self.user_map:
            return []
        u = self.user_map[user]
        scores = self.V @ self.U[u]
        top = np.argsort(-scores)[:n]
        return [(self.item_rev[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# UserKNN Recommender
# ---------------------------------------------------------------------------
class UserKNNRecommender(PersistentRecommenderMixin, BaseRecommender):
    def __init__(self, k=30):
        self.k = k

    def fit(self, ratings: pd.DataFrame):
        pivot = ratings.pivot(index='user_id', columns='item_id', values='rating')
        self.ratings = pivot
        sim = np.nan_to_num(np.corrcoef(np.nan_to_num(pivot.fillna(0)), rowvar=True))
        self.sim = sim
        self.users = pivot.index.tolist()
        self.items = pivot.columns.tolist()
        return self

    def predict(self, user, item):
        if user not in self.users or item not in self.items:
            return np.nan
        u_idx = self.users.index(user)
        neighbors = np.argsort(-self.sim[u_idx])[1:self.k+1]
        ratings = self.ratings.iloc[neighbors][item]
        sims = self.sim[u_idx, neighbors]
        return float(np.nansum(ratings * sims) / np.nansum(np.abs(sims)))

    def recommend(self, user, n=10):
        scores = {item: self.predict(user, item) for item in self.items}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:n]


# ---------------------------------------------------------------------------
# ItemKNN Recommender
# ---------------------------------------------------------------------------
class ItemKNNRecommender(PersistentRecommenderMixin, BaseRecommender):
    def __init__(self, k=30):
        self.k = k

    def fit(self, ratings: pd.DataFrame):
        pivot = ratings.pivot(index='item_id', columns='user_id', values='rating')
        self.ratings = pivot
        sim = np.nan_to_num(np.corrcoef(np.nan_to_num(pivot.fillna(0)), rowvar=True))
        self.sim = sim
        self.items = pivot.index.tolist()
        self.users = pivot.columns.tolist()
        return self

    def predict(self, user, item):
        if user not in self.users or item not in self.items:
            return np.nan
        i_idx = self.items.index(item)
        neighbors = np.argsort(-self.sim[i_idx])[1:self.k+1]
        ratings = self.ratings.iloc[neighbors][user]
        sims = self.sim[i_idx, neighbors]
        return float(np.nansum(ratings * sims) / np.nansum(np.abs(sims)))

    def recommend(self, user, n=10):
        scores = {item: self.predict(user, item) for item in self.items}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:n]

def register_and_save(name: str, model, path: str):
    """Registers a trained model and saves it to disk."""
    ModelRegistry.register(name, model)
    model.save(path)
    
def load_models():
  """Load all persisted models into the registry at startup."""
  # Example paths — adjust based on your project structure
  model_paths = {
    'baseline': 'models/baseline.pkl',
    'svd': 'models/svd.pkl',
    'user_knn': 'models/user_knn.pkl',
    'item_knn': 'models/item_knn.pkl'
    }

  for name, path in model_paths.items():
    try:
      model = PersistentRecommenderMixin.load(path)
      ModelRegistry.register(name, model)
    except Exception:
      pass