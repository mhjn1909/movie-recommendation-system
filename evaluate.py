import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import os

# ── α* = 0.30 (per paper Sec. III-J grid search) ─────────────────────────────
ALPHA = 0.30

# ── Temporal split cutoff: 2002-01-01 UTC (per paper Sec. III-G) ─────────────
CUTOFF_TIMESTAMP = 1009843200

# ── Load Data ────────────────────────────────────────────────────────
base = 'data'
movies_df = pd.read_csv(f'{base}/movies.dat', sep='::', engine='python',
                        names=['movieId','title','genres'], encoding='latin-1')
ratings_df = pd.read_csv(f'{base}/ratings.dat', sep='::', engine='python',
                         names=['userId','movieId','rating','timestamp'], encoding='latin-1')

print("Data loaded successfully!")
print(f"Total Movies  : {len(movies_df)}")
print(f"Total Ratings : {len(ratings_df)}")
print(f"Total Users   : {ratings_df['userId'].nunique()}")

# ── Temporal Train/Test Split (per paper Sec. III-G) ─────────────────────────
# Avoids data-leakage risk of random sampling (noted explicitly in paper)
train_data = ratings_df[ratings_df['timestamp'] <  CUTOFF_TIMESTAMP]
test_data  = ratings_df[ratings_df['timestamp'] >= CUTOFF_TIMESTAMP]

print(f"\nTemporal split at 2002-01-01")
print(f"Train size : {len(train_data)}")
print(f"Test size  : {len(test_data)}")

# ── Use full catalogue (3,706 movies, per paper Sec. III-G) ──────────────────
all_movie_ids = movies_df['movieId'].unique()
train_data = train_data[train_data['movieId'].isin(all_movie_ids)]
test_data  = test_data[test_data['movieId'].isin(all_movie_ids)]
movies_filtered = movies_df.copy().reset_index(drop=True)

# ── Build Rating Matrix ───────────────────────────────────────────────
train_matrix = train_data.pivot_table(
    index='movieId', columns='userId', values='rating').fillna(0)

# ── Cosine Similarity ─────────────────────────────────────────────────
print("\nBuilding cosine similarity matrix...")
cosine_sim = cosine_similarity(train_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim,
                              index=train_matrix.index,
                              columns=train_matrix.index)

# ── K-Means Clustering (K* = 10, per paper Sec. III-C) ───────────────────────
print("Running K-Means clustering...")
ALL_GENRES = ['Action','Adventure','Animation',"Children's",'Comedy','Crime',
              'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
              'Mystery','Romance','Sci-Fi','Thriller','War','Western']
movies_filtered['genre_list'] = movies_filtered['genres'].apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer(classes=ALL_GENRES)
genre_matrix = mlb.fit_transform(movies_filtered['genre_list'])
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
movies_filtered['cluster'] = kmeans.fit_predict(genre_matrix)

# ── Predict Rating Function ───────────────────────────────────────────
# Uses weighted mean of similar items' ratings (Eq. 3 in paper)
def predict_rating(userId, movieId):
    if movieId not in cosine_sim_df.index:
        return 3.0
    sim_scores = cosine_sim_df[movieId].drop(movieId)
    user_ratings = train_matrix.loc[
        train_matrix.index.isin(sim_scores.index), :]
    if userId not in user_ratings.columns:
        return 3.0
    user_col = user_ratings[userId]
    rated = user_col[user_col > 0]
    if len(rated) == 0:
        return 3.0
    sims = sim_scores[rated.index]
    if sims.sum() == 0:
        return 3.0
    predicted = np.dot(sims, rated) / sims.sum()
    return predicted

# ── Predict Rating with Cluster Boost (Eq. 4 in paper) ───────────────────────
# sim'(i,q) = sim(i,q) * (1 + α * δ(i,q))  — multiplicative, α* = 0.30
cluster_map = movies_filtered.set_index('movieId')['cluster'].to_dict()

def predict_rating_hybrid(userId, movieId):
    if movieId not in cosine_sim_df.index:
        return 3.0
    query_cluster = cluster_map.get(movieId, -1)
    sim_scores = cosine_sim_df[movieId].drop(movieId).copy()

    # Apply multiplicative cluster boost (Eq. 4)
    for mid in sim_scores.index:
        if cluster_map.get(mid, -2) == query_cluster:
            sim_scores[mid] *= (1 + ALPHA)

    user_ratings = train_matrix.loc[
        train_matrix.index.isin(sim_scores.index), :]
    if userId not in user_ratings.columns:
        return 3.0
    user_col = user_ratings[userId]
    rated = user_col[user_col > 0]
    if len(rated) == 0:
        return 3.0
    sims = sim_scores[rated.index]
    if sims.sum() == 0:
        return 3.0
    predicted = np.dot(sims, rated) / sims.sum()
    return predicted

# ── Calculate Metrics on FULL test set (per paper Sec. III-G revision) ────────
print(f"\nCalculating metrics on full test set ({len(test_data):,} samples)...")
print("This may take several minutes...")

actual_cf, predicted_cf = [], []
actual_hybrid, predicted_hybrid = [], []

for _, row in test_data.iterrows():
    uid, mid, rat = row['userId'], row['movieId'], row['rating']
    actual_cf.append(rat)
    predicted_cf.append(predict_rating(uid, mid))
    actual_hybrid.append(rat)
    predicted_hybrid.append(predict_rating_hybrid(uid, mid))

actual_cf       = np.array(actual_cf)
predicted_cf    = np.array(predicted_cf)
actual_hybrid   = np.array(actual_hybrid)
predicted_hybrid = np.array(predicted_hybrid)

# ── Regression Metrics ────────────────────────────────────────────────
rmse_hybrid   = np.sqrt(np.mean((actual_hybrid - predicted_hybrid)**2))
rmse_cf_only  = np.sqrt(np.mean((actual_cf    - predicted_cf)**2))      # actually measured
rmse_baseline = np.sqrt(np.mean((actual_hybrid - np.mean(actual_hybrid))**2))
mae_hybrid    = np.mean(np.abs(actual_hybrid - predicted_hybrid))

# ── Classification Metrics (Confusion Matrix) ─────────────────────────
threshold        = 3.5
actual_binary    = (actual_hybrid >= threshold).astype(int)
predicted_binary = (predicted_hybrid >= threshold).astype(int)

accuracy  = accuracy_score(actual_binary, predicted_binary)
precision = precision_score(actual_binary, predicted_binary)
recall    = recall_score(actual_binary, predicted_binary)
f1        = f1_score(actual_binary, predicted_binary)
cm        = confusion_matrix(actual_binary, predicted_binary)

# ── Print Results ─────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  EVALUATION RESULTS")
print(f"{'='*50}")
print(f"  RMSE  (Hybrid CF + Clustering) : {rmse_hybrid:.4f}")
print(f"  RMSE  (CF Only)                : {rmse_cf_only:.4f}")
print(f"  RMSE  (Baseline - Mean)        : {rmse_baseline:.4f}")
print(f"  MAE   (Hybrid)                 : {mae_hybrid:.4f}")
print(f"{'='*50}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"{'='*50}")

print("\nClassification Report:")
print(classification_report(actual_binary, predicted_binary,
      target_names=['Not Relevant', 'Relevant']))

print(f"\n{'='*55}")
print(f"  COMPARISON TABLE")
print(f"{'='*55}")
print(f"  {'Method':<30} {'RMSE':<10} {'MAE':<10}")
print(f"  {'-'*50}")
print(f"  {'Baseline (Mean Rating)':<30} {rmse_baseline:<10.4f} {np.mean(np.abs(actual_hybrid - np.mean(actual_hybrid))):<10.4f}")
print(f"  {'CF Only (No Clustering)':<30} {rmse_cf_only:<10.4f} {'N/A':<10}")
print(f"  {'Hybrid (CF + Clustering)':<30} {rmse_hybrid:<10.4f} {mae_hybrid:<10.4f}")
print(f"{'='*55}")

# ══ GRAPHS ══════════════════════════════════════════════════════════════

# ── Graph 1: RMSE Comparison ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
methods = ['Baseline\n(Mean)', 'CF Only', 'Hybrid\n(CF + Clustering)']
rmse_values = [rmse_baseline, rmse_cf_only, rmse_hybrid]
colors = ['#e74c3c', '#f39c12', '#2ecc71']
bars = ax.bar(methods, rmse_values, color=colors, width=0.45,
              edgecolor='white', linewidth=1.5)
ax.set_title('RMSE Comparison: Baseline vs CF vs Hybrid System',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('RMSE (Lower is Better)', fontsize=11)
ax.set_ylim(0, max(rmse_values) * 1.3)
for bar, val in zip(bars, rmse_values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom',
            fontweight='bold', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#f8f9fa')
fig.tight_layout()
plt.savefig('graph1_rmse_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: graph1_rmse_comparison.png")
plt.close()

# ── Graph 2: Rating Distribution ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
rating_counts = ratings_df['rating'].value_counts().sort_index()
ax.bar(rating_counts.index, rating_counts.values,
       color='#3498db', edgecolor='white', linewidth=1.2)
ax.set_title('Rating Distribution in MovieLens Dataset',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Rating (1-5 Stars)', fontsize=11)
ax.set_ylabel('Number of Ratings', fontsize=11)
ax.set_xticks([1, 2, 3, 4, 5])
for x, y in zip(rating_counts.index, rating_counts.values):
    ax.text(x, y + 5000, f'{y:,}', ha='center', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#f8f9fa')
fig.tight_layout()
plt.savefig('graph2_rating_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: graph2_rating_distribution.png")
plt.close()

# ── Graph 3: Cluster Distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
cluster_counts = movies_filtered['cluster'].value_counts().sort_index()
ax.bar([f'Cluster {i}' for i in cluster_counts.index],
       cluster_counts.values,
       color='#9b59b6', edgecolor='white', linewidth=1.2)
ax.set_title('Movie Distribution Across K-Means Clusters',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel('Number of Movies', fontsize=11)
plt.xticks(rotation=45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#f8f9fa')
fig.tight_layout()
plt.savefig('graph3_cluster_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: graph3_cluster_distribution.png")
plt.close()

# ── Graph 4: Actual vs Predicted ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(actual_hybrid, predicted_hybrid, alpha=0.4, color='#1abc9c',
           edgecolors='white', linewidth=0.5, s=30)
ax.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_title('Actual vs Predicted Ratings',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Actual Rating', fontsize=11)
ax.set_ylabel('Predicted Rating', fontsize=11)
ax.legend(fontsize=10)
ax.set_facecolor('#f8f9fa')
fig.tight_layout()
plt.savefig('graph4_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print("Saved: graph4_actual_vs_predicted.png")
plt.close()

# ── Graph 5: Confusion Matrix ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Relevant', 'Relevant'],
            yticklabels=['Not Relevant', 'Relevant'],
            linewidths=2, linecolor='white',
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title('Confusion Matrix\n(Threshold: Rating >= 3.5 = Relevant)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
fig.tight_layout()
plt.savefig('graph5_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: graph5_confusion_matrix.png")
plt.close()

# ── Graph 6: Precision Recall F1 Bar Chart ───────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors2 = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
bars2 = ax.bar(metrics_names, metrics_values, color=colors2,
               width=0.45, edgecolor='white', linewidth=1.5)
ax.set_title('Classification Metrics of Hybrid System',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Score (0 to 1)', fontsize=11)
ax.set_ylim(0, 1.2)
for bar, val in zip(bars2, metrics_values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', va='bottom',
            fontweight='bold', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#f8f9fa')
fig.tight_layout()
plt.savefig('graph6_classification_metrics.png', dpi=150, bbox_inches='tight')
print("Saved: graph6_classification_metrics.png")
plt.close()

print("\n" + "="*50)
print("  ALL DONE! 6 graphs saved successfully.")
print("  Add them to your research paper!")
print("="*50)
