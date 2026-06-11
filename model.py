import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from rapidfuzz import process, fuzz
import os
import re

# ─── GENRES LIST (MovieLens uses | separator) ───────────────────────────────
ALL_GENRES = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

# ─── Boost parameter α* = 0.30 (chosen by grid search, per paper Sec. III-J) ─
ALPHA = 0.30

# ─── Fuzzy match threshold = 0.75 (normalised Levenshtein, per paper Sec. III-B)
FUZZ_THRESHOLD = 60.0  # rapidfuzz uses 0–100 scale


def load_data():
    """
    Load MovieLens 1M dataset.
    Files expected in data/ folder:
      - movies.dat  (MovieID::Title::Genres)
      - ratings.dat (UserID::MovieID::Rating::Timestamp)
    """
    base = os.path.join(os.path.dirname(__file__), 'data')

    # ── Load movies ──────────────────────────────────────────────────────────
    movies_df = pd.read_csv(
        os.path.join(base, 'movies.dat'),
        sep='::',
        engine='python',
        names=['movieId', 'title', 'genres'],
        encoding='latin-1'
    )

    # ── Load ratings ─────────────────────────────────────────────────────────
    ratings_df = pd.read_csv(
        os.path.join(base, 'ratings.dat'),
        sep='::',
        engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp'],
        encoding='latin-1'
    )

    # ── Build genre feature matrix for K-Means ───────────────────────────────
    movies_df['genre_list'] = movies_df['genres'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer(classes=ALL_GENRES)
    genre_matrix = mlb.fit_transform(movies_df['genre_list'])
    genre_df = pd.DataFrame(genre_matrix, columns=ALL_GENRES)

    # ── K-Means Clustering (K* = 10, per paper Sec. III-C) ───────────────────
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    movies_df['cluster'] = kmeans.fit_predict(genre_df)

    # ── Temporal train/test split (per paper Sec. III-G) ─────────────────────
    # Training: timestamps before 1 Jan 2002 (~80%)
    # All ratings used for the similarity matrix (training portion only)
    CUTOFF_TIMESTAMP = 1009843200  # Unix timestamp for 2002-01-01 00:00:00 UTC
    ratings_train = ratings_df[ratings_df['timestamp'] < CUTOFF_TIMESTAMP]

    # ── Build user-item rating matrix from ALL 3,706 movies ──────────────────
    # (paper uses full catalogue, not a top-N subset)
    rating_matrix = ratings_train.pivot_table(
        index='movieId',
        columns='userId',
        values='rating'
    ).fillna(0)

    # ── Item-Based Collaborative Filtering (Cosine Similarity) ───────────────
    cosine_sim = cosine_similarity(rating_matrix)
    cosine_sim_df = pd.DataFrame(
        cosine_sim,
        index=rating_matrix.index,
        columns=rating_matrix.index
    )

    # ── Filter movies_df to only movies present in rating matrix ─────────────
    movies_df = movies_df[movies_df['movieId'].isin(rating_matrix.index)].reset_index(drop=True)

    # ── Index map: title → position ──────────────────────────────────────────
    indices = pd.Series(movies_df.index, index=movies_df['title'])

    return movies_df, ratings_df, cosine_sim_df, indices

def clean_title(title):
    # Remove year (2012)
    title = re.sub(r"\(\d{4}\)", "", title)
    
    
    if "," in title:
        parts = title.split(",")
        title = parts[1].strip() + " " + parts[0].strip()
    
    return title.strip().lower()


def find_closest_movie(movie_input, movies_df):
    titles = movies_df['title'].tolist()

    # Clean titles
    cleaned_titles = [clean_title(t) for t in titles]
    clean_input = movie_input.lower()

    # Exact match after cleaning
    if clean_input in cleaned_titles:
        idx = cleaned_titles.index(clean_input)
        return titles[idx]

    # Fuzzy match (better)
    result = process.extractOne(
        clean_input,
        cleaned_titles,
        scorer=fuzz.token_set_ratio,
        score_cutoff=50
    )

    if result:
        return titles[result[2]]  # return original title

    return None

def get_recommendations(movie_input, movies_df, cosine_sim_df, indices, n=10):
    """
    Get top-n recommendations for a given movie using:
    1. Item-Based Collaborative Filtering (cosine similarity on ratings)
    2. K-Means Cluster-Aware Similarity Boosting (multiplicative, α* = 0.30)

    Boost formula (Eq. 4 in paper):
        sim'(i, q) = sim(i, q) * (1 + α * δ(i, q))
    where δ(i, q) = 1 if items i and q share the same K-Means cluster, else 0.
    """
    matched_title = find_closest_movie(movie_input, movies_df)
    if matched_title is None:
        return None

    # Get movieId of matched title
    movie_row = movies_df[movies_df['title'] == matched_title].iloc[0]
    movie_id = movie_row['movieId']
    input_cluster = movie_row['cluster']

    if movie_id not in cosine_sim_df.index:
        return None

    # Get similarity scores for this movie vs all others
    sim_scores = cosine_sim_df[movie_id].drop(movie_id)
    sim_df = pd.DataFrame({'movieId': sim_scores.index, 'similarity': sim_scores.values})

    # Merge with movie info
    sim_df = sim_df.merge(movies_df[['movieId', 'title', 'genres', 'cluster']], on='movieId')

    # ── Cluster-Aware Similarity Boosting (multiplicative, Eq. 4) ────────────
    # sim'(i,q) = sim(i,q) * (1 + α * 1[C(i)==C(q)])
    sim_df['delta'] = (sim_df['cluster'] == input_cluster).astype(float)
    sim_df['final_score'] = sim_df['similarity'] * (1 + ALPHA * sim_df['delta'])

    # Sort and take top n
    sim_df = sim_df.sort_values('final_score', ascending=False).head(n).reset_index(drop=True)
    sim_df['rank'] = sim_df.index + 1
    sim_df['similarity'] = sim_df['final_score'].clip(0, 1)

    return matched_title, sim_df[['rank', 'title', 'genres', 'similarity', 'cluster']]


def get_movie_clusters(movies_df):
    """Return cluster summary stats."""
    return movies_df.groupby('cluster').agg(
        count=('title', 'count'),
        genres=('genres', lambda x: x.mode()[0] if len(x) > 0 else '')
    ).reset_index()
