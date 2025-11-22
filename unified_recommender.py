"""
Unified Recommendation Models Module
Includes:
- BaselineBias
- SVDRecommender
- UserKNN
- ItemKNN
- evaluate() function

This file is intended to be imported by the benchmark runner.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------------------------
# Automatic model registration + saving after training
# ---------------------------------------------------------------------------

class PersistentRecommenderMixin:
    """Adds save/load behavior to any recommender."""
    def save(self, path: str):
        ModelIO.save(self, path)

    @classmethod
    def load(cls, path: str):
        return ModelIO.load(path)
    
# ----------------------------------------------------
# Baseline Bias Model
# ----------------------------------------------------
class BaselineBias(PersistentRecommenderMixin):
    def fit(self, df, n_users=None, n_items=None):
        self.global_mean = df["rating"].mean()
        self.user_bias = df.groupby("user_idx")["rating"].mean() - self.global_mean
        self.item_bias = df.groupby("item_idx")["rating"].mean() - self.global_mean
        return self

    def predict(self, u, i):
        bu = self.user_bias.get(u, 0)
        bi = self.item_bias.get(i, 0)
        return np.clip(self.global_mean + bu + bi, 1.0, 5.0)

# ----------------------------------------------------
# SVD Recommender
# ----------------------------------------------------
class SVDRecommender(PersistentRecommenderMixin):
  def __init__(self, n_factors=50, lr=0.005, reg=0.02, n_epochs=20):
      self.n_factors = n_factors
      self.lr = lr
      self.reg = reg
      self.n_epochs = n_epochs

  def fit(self, df, n_users, n_items):
      self.global_mean = df["rating"].mean()

      self.bu = np.zeros(n_users)
      self.bi = np.zeros(n_items)

      self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
      self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

      for _ in range(self.n_epochs):
          for _, row in df.iterrows():
              u = row.user_idx
              i = row.item_idx
              r = row.rating

              pred = self.predict(u, i)
              err = r - pred

              self.bu[u] += self.lr * (err - self.reg * self.bu[u])
              self.bi[i] += self.lr * (err - self.reg * self.bi[i])

              self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
              self.Q[i] += self.lr * (err * self.P[u] - self.reg * self.Q[i])
      return self
  
  def predict(self, u, i):
      dot = np.dot(self.P[u], self.Q[i])
      return np.clip(self.global_mean + self.bu[u] + self.bi[i] + dot, 1.0, 5.0)

# ----------------------------------------------------
# User-Based KNN Recommender
# ----------------------------------------------------
class UserKNN(PersistentRecommenderMixin):
    def __init__(self, k=40, similarity="cosine"):
        self.k = k
        self.similarity = similarity

    def fit(self, df, n_users, n_items):
        R = np.zeros((n_users, n_items))
        for row in df.itertuples():
            R[row.user_idx, row.item_idx] = row.rating
        self.R = R

        if self.similarity == "cosine":
            self.sim_matrix = cosine_similarity(R)
        elif self.similarity == "pearson":
            self.sim_matrix = np.corrcoef(R)
            self.sim_matrix = np.nan_to_num(self.sim_matrix)
        else:
            raise ValueError("Unknown similarity metric")

        return self

    def predict(self, u, i):
        sims = self.sim_matrix[u]
        item_ratings = self.R[:, i]

        idx = np.argsort(sims)[::-1][:self.k]
        rated = item_ratings[idx] > 0

        if rated.sum() == 0:
            user_ratings = self.R[u]
            if np.sum(user_ratings > 0) > 0:
                return np.nanmean(user_ratings[user_ratings > 0])
            return 3.0

        return np.mean(item_ratings[idx][rated])

# ----------------------------------------------------
# Item-Based KNN Recommender
# ----------------------------------------------------
class ItemKNN(PersistentRecommenderMixin):
    def __init__(self, k=40, similarity="cosine"):
        self.k = k
        self.similarity = similarity

    def fit(self, df, n_users, n_items):
        R = np.zeros((n_users, n_items))
        for row in df.itertuples():
            R[row.user_idx, row.item_idx] = row.rating
        self.R = R

        R_T = R.T

        if self.similarity == "cosine":
            self.sim_matrix = cosine_similarity(R_T)
        elif self.similarity == "pearson":
            self.sim_matrix = np.corrcoef(R_T)
            self.sim_matrix = np.nan_to_num(self.sim_matrix)
        else:
            raise ValueError("Unknown similarity metric")

        return self

    def predict(self, u, i):
        sims = self.sim_matrix[i]
        user_ratings = self.R[u]

        idx = np.argsort(sims)[::-1][:self.k]
        rated = user_ratings[idx] > 0

        if rated.sum() == 0:
            user_ratings_nonzero = user_ratings[user_ratings > 0]
            if len(user_ratings_nonzero) > 0:
                return np.nanmean(user_ratings_nonzero)
            return 3.0

        return np.mean(user_ratings[idx][rated])

# ----------------------------------------------------
# Unified Evaluate Function
# ----------------------------------------------------

def evaluate(model, train_df, test_df, n_users, n_items):
    model.fit(train_df, n_users, n_items)

    preds = []
    for row in test_df.itertuples():
        preds.append(model.predict(int(row.user_idx), int(row.item_idx)))

    return model , np.sqrt(mean_squared_error(test_df["rating"], preds))

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, model):
        cls._registry[name] = model

    @classmethod
    def get(cls, name: str):
        return cls._registry.get(name)

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())
    
# ---------------------------------------------------------------------------
# Model Persistence Utilities
# ---------------------------------------------------------------------------
import pickle

class ModelIO:
    @staticmethod
    def save(model, path: str):
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
        

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, model):
        cls._registry[name] = model

    @classmethod
    def get(cls, name: str):
        return cls._registry.get(name)

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())

# ---------------------------------------------------------------------------
    
def register_and_save(name: str, model, path: str):
    """Registers a trained model and saves it to disk."""
    ModelRegistry.register(name, model)
    model.save(path)
    
"""
Benchmark Runner for Evaluating All Models
"""

if __name__ == "__main__":
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Get CWD and Load MovieLens 100k
    dir = os.getcwd()
    df = pd.read_csv(dir+"/data/raw/ml-100k/u.data", 
                     sep="\t", 
                     names=["user", "item", "rating", "timestamp"])
    df = df.drop(columns=["timestamp"])
    df.head()

    # Reindex users and items
    df["user_idx"], _ = pd.factorize(df["user"])
    df["item_idx"], _ = pd.factorize(df["item"])

    n_users = df["user_idx"].nunique()
    n_items = df["item_idx"].nunique()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    models = [
        ("BaselineBias", BaselineBias()),
        ("SVD", SVDRecommender()),
        ("UserKNN", UserKNN(k=60, similarity="cosine")),
        ("ItemKNN", ItemKNN(k=60, similarity="pearson")),
    ]

    print("-------------------------------")
    print(" Recommender Performance (RMSE)")
    print("-------------------------------")

    trained_models = []
    names = []
    for name, model in models:
        tmodel, rmse = evaluate(model, train_df, test_df, n_users, n_items)
        print(f"{name:<14}: {rmse:.4f}")
        register_and_save(name, tmodel, "models/" + name + ".pkl")