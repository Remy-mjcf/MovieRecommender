# train_models.py
"""
Train all recommenders using MovieLens 100k dataset,
register them in the ModelRegistry, and save each model for deployment.
"""

import os
import pandas as pd
from recommender_models import (
    BaselineBias,
    SVDRecommender,
    UserKNNRecommender,
    ItemKNNRecommender,
    ModelRegistry,
    register_and_save
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data/raw/ml-100k")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load MovieLens 100k
# ------------------------------------------------------------
columns = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv(os.path.join(DATA_DIR, "u.data"), sep='\t', names=columns)
df = df.drop(columns=["timestamp"])

# ------------------------------------------------------------
# Train and Save Models
# ------------------------------------------------------------

# 1. Baseline Bias
baseline = BaselineBias().fit(df)
register_and_save("baseline", baseline, os.path.join(MODEL_DIR, "baseline.pkl"))
print("BaselineBias trained and saved.")

# 2. SVD
svd = SVDRecommender(n_factors=20, n_iters=20).fit(df)
register_and_save("svd", svd, os.path.join(MODEL_DIR, "svd.pkl"))
print("SVDRecommender trained and saved.")

# 3. UserKNN
user_knn = UserKNNRecommender(k=60).fit(df)
register_and_save("user_knn", user_knn, os.path.join(MODEL_DIR, "user_knn.pkl"))
print("UserKNNRecommender trained and saved.")

# 4. ItemKNN
item_knn = ItemKNNRecommender(k=60).fit(df)
register_and_save("item_knn", item_knn, os.path.join(MODEL_DIR, "item_knn.pkl"))
print("ItemKNNRecommender trained and saved.")

print("All models trained, registered, and saved successfully.")
