import streamlit as st
import pandas as pd
import numpy as np
from model import load_data, get_recommendations, get_movie_clusters

# Page config
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

.main { background-color: #0a0a0f; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #12121f 50%, #0a0f1a 100%);
    color: #e8e8f0;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f5c842, #ff6b6b, #c678dd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
}

.hero-sub {
    text-align: center;
    color: #888;
    font-size: 1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.movie-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: all 0.2s;
}

.movie-card:hover {
    background: rgba(245, 200, 66, 0.08);
    border-color: rgba(245, 200, 66, 0.3);
}

.rank-badge {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #f5c842;
    min-width: 2rem;
    font-weight: 700;
}

.movie-title {
    font-size: 1rem;
    font-weight: 500;
    color: #e8e8f0;
}

.movie-genre {
    font-size: 0.78rem;
    color: #888;
    margin-top: 2px;
}

.similarity-bar {
    height: 4px;
    background: linear-gradient(90deg, #f5c842, #ff6b6b);
    border-radius: 2px;
    margin-top: 6px;
}

.cluster-badge {
    background: rgba(198, 120, 221, 0.15);
    border: 1px solid rgba(198, 120, 221, 0.3);
    color: #c678dd;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
}

.section-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555;
    margin: 1.5rem 0 0.8rem 0;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-size: 1rem !important;
    padding: 0.8rem 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(245, 200, 66, 0.5) !important;
    box-shadow: 0 0 0 2px rgba(245, 200, 66, 0.1) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #f5c842, #ff6b6b) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.88 !important;
}

.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-box {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}

.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #f5c842;
}

.stat-label {
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2rem 0;
}

.not-found {
    background: rgba(255, 107, 107, 0.08);
    border: 1px solid rgba(255, 107, 107, 0.2);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #ff6b6b;
    font-size: 0.9rem;
}

.suggestion-chip {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #aaa;
    margin: 3px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    return load_data()


# Load data
with st.spinner("Loading movie database..."):
    movies_df, ratings_df, cosine_sim, indices = load()

# Hero section
st.markdown('<div class="hero-title">CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Intelligent Movie Recommendation System</div>', unsafe_allow_html=True)

# Stats
total_movies = len(movies_df)
total_ratings = len(ratings_df)
st.markdown(f"""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-num">{total_movies:,}</div>
        <div class="stat-label">Movies</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">{total_ratings:,}</div>
        <div class="stat-label">Ratings</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">K-Means</div>
        <div class="stat-label">Clustering</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">Item-CF</div>
        <div class="stat-label">Algorithm</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Search
st.markdown('<div class="section-label">🎬 Enter a movie you like</div>', unsafe_allow_html=True)
movie_input = st.text_input("", placeholder="e.g. Toy Story, Jumanji, GoodFellas...", label_visibility="collapsed")

# Quick suggestions
st.markdown("""
<div style="margin-bottom:1.2rem">
    <span style="font-size:0.75rem;color:#555;text-transform:uppercase;letter-spacing:0.1em;">Try: </span>
    <span class="suggestion-chip">Toy Story</span>
    <span class="suggestion-chip">GoodFellas</span>
    <span class="suggestion-chip">Fargo</span>
    <span class="suggestion-chip">Silence of the Lambs</span>
    <span class="suggestion-chip">Pulp Fiction</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_btn = st.button("✨ Get Recommendations")

# Results
if search_btn and movie_input:
    with st.spinner("Finding your perfect matches..."):
        results = get_recommendations(movie_input, movies_df, cosine_sim, indices, n=10)

    if results is None:
        st.markdown(f"""
        <div class="not-found">
            ⚠️ Movie "<b>{movie_input}</b>" not found in database. 
            Try a different spelling or one of the suggestions above.
        </div>
        """, unsafe_allow_html=True)
    else:
        matched_movie, recommendations = results
        st.markdown(f'<div class="section-label">✅ Matched: {matched_movie}</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🎯 Top Recommendations</div>', unsafe_allow_html=True)

        for i, row in recommendations.iterrows():
            sim_width = int(row['similarity'] * 100)
            st.markdown(f"""
            <div class="movie-card">
                <div class="rank-badge">{row['rank']}</div>
                <div style="flex:1">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-genre">{row['genres']}</div>
                    <div class="similarity-bar" style="width:{sim_width}%"></div>
                </div>
                <div>
                    <span class="cluster-badge">Cluster {row['cluster']}</span>
                    <div style="text-align:right;font-size:0.78rem;color:#666;margin-top:4px">{row['similarity']:.0%} match</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

elif search_btn and not movie_input:
    st.warning("Please enter a movie name first.")

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#333;font-size:0.78rem">
    Built with Item-Based Collaborative Filtering + K-Means Clustering · MovieLens Dataset · IILM University Minor Project 2025-26
</div>
""", unsafe_allow_html=True)
