# app.py
"""
Streamlit MovieLens Recommender App (Enhanced)
Features:
- Loads persisted models into ModelRegistry
- Model selection and predictions
- Top-N recommendations with movie metadata (title, year, genres)
- Tabs-based UI and improved UX
- Caching for loaded models
- Embedding visualization (PCA)
- Retrain selected model from uploaded CSV or re-train on the original ML-100k
- Save/re-register retrained models

Notes:
- Assumes recommender_models.py is importable and model pickles live in ./models/
- Assumes MovieLens ml-100k data is in ./ml-100k/
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import traceback
from sklearn.decomposition import PCA
import altair as alt

from recommender_models import ModelIO, ModelRegistry, BaselineBias, SVDRecommender, \
    UserKNNRecommender, ItemKNNRecommender

# ---------------------------
# Config
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data/raw/ml-100k")
MOVIES_PATH = os.path.join(DATA_DIR, "u.item")

st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource
def load_movies():
    """
    Load MovieLens ml-100k `u.item` file and return a dataframe with:
      - movie_id (int)
      - title (str)
      - year (int or None)
      - genres (list of genre names)
    """
    import os

    MOVIES_PATH = os.path.join(DATA_DIR, "u.item")
    if not os.path.exists(MOVIES_PATH):
        st.warning(f"Movie metadata file not found at {MOVIES_PATH}")
        return pd.DataFrame(columns=["movie_id", "title", "year", "genres"])

    # Official MovieLens 100k genre names (19)
    genre_names = [
        "Unknown","Action","Adventure","Animation","Children","Comedy","Crime",
        "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
        "Romance","Sci-Fi","Thriller","War","Western"
    ]

    # Build column names: first five fields then 19 genre flag columns
    cols = ["movie_id", "title", "release_date", "video_release", "imdb_url"] + \
           [f"g{i}" for i in range(len(genre_names))]

    # Read file (u.item uses '|' separator and latin-1 encoding)
    df = pd.read_csv(MOVIES_PATH, sep="|", names=cols, encoding="latin-1", header=None)

    # movie_id should be int
    df["movie_id"] = df["movie_id"].astype(int)

    # extract year safely from release_date (format: dd-MMM-yyyy or empty)
    def extract_year(x):
        try:
            if pd.isna(x) or str(x).strip() == "":
                return None
            # Many entries look like "01-Jan-1995"
            parts = str(x).split("-")
            year = int(parts[-1])
            return year
        except Exception:
            return None

    df["year"] = df["release_date"].apply(extract_year)

    # Build genre lists from flag columns and map to names
    genre_cols = [f"g{i}" for i in range(len(genre_names))]

    def genres_row(row):
        return [genre_names[i] for i, c in enumerate(genre_cols) if int(row[c]) == 1]

    df["genres"] = df.apply(genres_row, axis=1)

    # Keep only fields we need
    return df[["movie_id", "title", "year", "genres"]]


@st.cache_resource
def discover_and_load_models():
    # scan model dir and try to load pickles, register into ModelRegistry
    if not os.path.exists(MODEL_DIR):
        return []
    loaded = []
    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith('.pkl'):
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(MODEL_DIR, fname)
        try:
            obj = ModelIO.load(path)
            ModelRegistry.register(name, obj)
            loaded.append(name)
        except Exception as e:
            print(f'Failed loading {fname}: {e}')
            traceback.print_exc()
    return loaded

# convenience: safe loader
@st.cache_resource
def load_model_by_name(name: str):
    # If model already in registry, return it
    m = ModelRegistry.get(name)
    if m is not None:
        return m
    # else try loading from file
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if os.path.exists(path):
        m = ModelIO.load(path)
        ModelRegistry.register(name, m)
        return m
    return None

# ---------------------------
# UI: Header and load movies/models
# ---------------------------
st.title("🎬 MovieLens Recommender — Enhanced")
movies = load_movies()
loaded_models = discover_and_load_models()

# Sidebar: model ops
st.sidebar.header("Models & Actions")
model_names = ModelRegistry.list_models()
if len(model_names)==0:
    st.sidebar.warning("No models found in registry. Run train_models.py to create models.")

selected_model_name = st.sidebar.selectbox("Choose a model", model_names)
selected_model = None
if selected_model_name:
    selected_model = load_model_by_name(selected_model_name)
    if selected_model is None:
        st.sidebar.error("Failed to load selected model.")

# Retrain controls
st.sidebar.markdown("---")
st.sidebar.header("Retrain / Upload")
retrain_source = st.sidebar.radio("Retrain from:", ("ML-100k", "Upload CSV"))
upload_file = None
if retrain_source == "Upload CSV":
    upload_file = st.sidebar.file_uploader("Upload ratings CSV", type=['csv'])
retrain_button = st.sidebar.button("Retrain selected model")

# Save new model name
save_name = st.sidebar.text_input("Save model as (name)", value=f"{selected_model_name}_v2")
save_button = st.sidebar.button("Save & Register")

# ---------------------------
# Main Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Predict", "Top-N", "Diagnostics", "Retrain", "User Profile"])

with tab1:
    st.header("Predict a Single Rating")
    col1, col2 = st.columns([1,3])
    with col1:
        uid = st.number_input("User ID", min_value=1, step=1, value=1)
        movie_sel = st.selectbox("Movie", movies['title'].tolist())
        mid = int(movies.loc[movies['title']==movie_sel,'movie_id'].iloc[0])
        pred_button = st.button("Predict")
    with col2:
        st.write("### Movie info")
        st.write(movies.loc[movies['title']==movie_sel].iloc[0].to_dict())

    if pred_button:
        if selected_model is None:
            st.error("No model loaded")
        else:
            try:
                # models expect raw ids (depending on implementation); try string then int
                p = selected_model.predict(uid, mid)
                st.metric("Predicted rating", f"{p:.3f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                traceback.print_exc()

with tab2:
    st.header("Top-N Recommendations")
    uid_rec = st.number_input("User ID for recommendations", min_value=1, step=1, value=1, key='rec_user')
    top_n = st.slider("Top N", 5, 50, 10)
    filter_genre = st.multiselect("Filter genres (optional)", sorted({g for sub in movies['genres'] for g in sub}))
    st.subheader("Top-N by Genre")
    genre_for_top = st.selectbox("Genre for Personalized Top-N", sorted({g for sub in movies['genres'] for g in sub}))

    if st.button("Get Recommendations", key='rec_button'):
        if selected_model is None:
            st.error("No model loaded")
        else:
            try:
                recs = selected_model.recommend(uid_rec, n=top_n)
                recs_df = pd.DataFrame(recs, columns=['movie_id','score']).merge(movies, on='movie_id', how='left')
                if filter_genre:
                    recs_df = recs_df[recs_df['genres'].apply(lambda g: any(x in g for x in filter_genre))]
                st.write("### Filtered Recommendations")
                st.dataframe(recs_df[['title','year','genres','score']].head(top_n))

                genre_specific = recs_df[recs_df['genres'].apply(lambda g: genre_for_top in g)]
                st.write(f"### Top {top_n} in {genre_for_top}")
                st.dataframe(genre_specific[['title','year','genres','score']].head(top_n))
            except Exception as e:
                st.error(f"Recommendation error: {e}")
                traceback.print_exc()
    st.header("Top-N Recommendations")
    uid_rec = st.number_input("User ID for recommendations", min_value=1, step=1, value=1, key='rec_user2')
    top_n = st.slider("Top-N", 5, 50, 10)
    filter_genre = st.multiselect("Filter your genres (optional)", sorted({g for sub in movies['genres'] for g in sub}))
    if st.button("Get Recommendations", key='rec_button2'):
        if selected_model is None:
            st.error("No model loaded")
        else:
            try:
                recs = selected_model.recommend(uid_rec, n=top_n)
                recs_df = pd.DataFrame(recs, columns=['movie_id','score'])
                recs_df = recs_df.merge(movies, on='movie_id', how='left')
                if filter_genre:
                    recs_df = recs_df[recs_df['genres'].apply(lambda g: any(x in g for x in filter_genre))]
                st.dataframe(recs_df[['title','year','score']].head(top_n))
            except Exception as e:
                st.error(f"Recommendation error: {e}")
                traceback.print_exc()

with tab3:
    st.header("Diagnostics & Visualization")
    st.write("Model list:", ModelRegistry.list_models())
    st.write("Loaded model:", selected_model_name)

    ratings_path = os.path.join(DATA_DIR, 'u.data')
    ratings_df = pd.read_csv(ratings_path, sep='	', names=['user_id','item_id','rating','timestamp'])

    st.subheader("Rating Distribution")
    hist = alt.Chart(ratings_df).mark_bar().encode(x=alt.X('rating:Q', bin=True), y='count()')
    st.altair_chart(hist, use_container_width=True)

    st.subheader("User Similarity Heatmap (Sample 200)")
    sample_users = ratings_df['user_id'].unique()[:200]
    pivot = ratings_df[ratings_df['user_id'].isin(sample_users)].pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    sim = np.corrcoef(pivot)
    sim_df = pd.DataFrame(sim, index=sample_users, columns=sample_users)
    st.dataframe(sim_df)

    st.subheader("Embedding Visualization")
    st.header("Diagnostics & Visualization")
    st.write("Model list:", ModelRegistry.list_models())
    st.write("Loaded model:", selected_model_name)
    if selected_model is not None:
        st.subheader("Model details")
        st.write(type(selected_model))

        if hasattr(selected_model, 'U') and hasattr(selected_model, 'V'):
            st.write("Visualizing user/item embeddings (PCA)...")
            # build embeddings
            users_emb = selected_model.U if hasattr(selected_model, 'U') else None
            items_emb = selected_model.V if hasattr(selected_model, 'V') else None
            if users_emb is not None:
                pca = PCA(n_components=2)
                up = pca.fit_transform(users_emb)
                dfu = pd.DataFrame(up, columns=['x','y'])
                dfu['type'] = 'user'
                st.altair_chart(alt.Chart(dfu.sample(min(500,len(dfu)))).mark_circle(size=40).encode(x='x',y='y'), use_container_width=True)
            if items_emb is not None:
                pca = PCA(n_components=2)
                ip = pca.fit_transform(items_emb)
                dfi = pd.DataFrame(ip, columns=['x','y'])
                dfi['type'] = 'item'
                st.altair_chart(alt.Chart(dfi.sample(min(500,len(dfi)))).mark_circle(size=40).encode(x='x',y='y'), use_container_width=True)

with tab4:
    st.header("Retrain / Upload Data")
    st.write("Retrain source:", retrain_source)
    if retrain_source == 'Upload CSV' and upload_file is not None:
        st.write("Preview uploaded data:")
        try:
            up_df = pd.read_csv(upload_file)
            st.dataframe(up_df.head())
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

    if retrain_button:
        if selected_model is None:
            st.error("Select a model first")
        else:
            # build training dataframe
            if retrain_source == 'ML-100k':
                ratings = pd.read_csv(os.path.join(DATA_DIR, 'u.data'), sep='	', names=['user_id','item_id','rating','timestamp']).drop(columns=['timestamp'])
            else:
                try:
                    ratings = pd.read_csv(upload_file)
                except Exception as e:
                    st.error(f"Failed to read upload: {e}")
                    ratings = None
            if ratings is not None:
                st.info("Training... this may take a moment")
                try:
                    # call fit on a new instance of same class
                    cls = type(selected_model)
                    inst = cls()
                    inst.fit(ratings)
                    st.success("Retraining finished")
                    # assign to selected_model for immediate use
                    ModelRegistry.register(save_name, inst)
                except Exception as e:
                    st.error(f"Retrain failed: {e}")
                    traceback.print_exc()

    if save_button:
        if save_name.strip()=="":
            st.error("Provide a valid name to save model")
        else:
            m = ModelRegistry.get(save_name)
            if m is None:
                st.error("No retrained model available under that name. Retrain first.")
            else:
                path = os.path.join(MODEL_DIR, f"{save_name}.pkl")
                try:
                    ModelIO.save(m, path)
                    st.success(f"Saved model to {path}")
                except Exception as e:
                    st.error(f"Failed to save model: {e}")
                    traceback.print_exc()

with tab5:
    st.header("User Profile & Ratings")
    user_prof_id = st.number_input("User ID", min_value=1, step=1, value=1, key='user_prof')

    ratings_path = os.path.join(DATA_DIR, 'u.data')
    ratings_df = pd.read_csv(ratings_path, sep='	', names=['user_id','item_id','rating','timestamp'])

    user_r = ratings_df[ratings_df['user_id'] == user_prof_id]

    if user_r.empty:
        st.warning("No ratings found for this user.")
    else:
        merged = user_r.merge(movies, left_on='item_id', right_on='movie_id', how='left')
        st.write(f"### Ratings for User {user_prof_id}")
        st.dataframe(merged[['title','year','genres','rating']])

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Models found:")
for nm in ModelRegistry.list_models():
    st.sidebar.write(nm)
