"""
=============================================================================
Thiranex Internship - Project 2: Customer Segmentation Project
=============================================================================
Author: Intern
Dataset: Mall Customers Dataset
Task: Segment customers based on demographic and purchase behavior using
      unsupervised machine learning (K-Means, Hierarchical Clustering).
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import json

# Setup output folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# Plot styling configuration
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300


def load_and_inspect_data():
    csv_path = os.path.join(DATA_DIR, "Mall_Customers.csv")
    print(f"[1/7] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    df.columns = ["CustomerID", "Gender", "Age", "Annual_Income_k$", "Spending_Score"]
    
    print("\n--- Dataset Summary ---")
    print(f"Total Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    return df


def perform_eda(df):
    print("\n[2/7] Generating Exploratory Data Analysis (EDA) Visualizations...")
    
    # 1. Distribution of Numerical Features
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    features = ["Age", "Annual_Income_k$", "Spending_Score"]
    colors = ["#4361ee", "#2ec4b6", "#e71d36"]
    
    for ax, feat, col in zip(axes, features, colors):
        sns.histplot(df[feat], kde=True, ax=ax, color=col, bins=15, edgecolor="black", alpha=0.6)
        ax.axvline(df[feat].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {df[feat].mean():.1f}")
        ax.axvline(df[feat].median(), color="darkblue", linestyle=":", linewidth=1.5, label=f"Median: {df[feat].median():.1f}")
        ax.set_title(f"Distribution of {feat.replace('_', ' ')}", fontsize=13, weight="bold")
        ax.set_xlabel(feat.replace('_', ' '), fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend()
    
    plt.tight_layout()
    dist_path = os.path.join(VIS_DIR, "01_eda_distributions.png")
    plt.savefig(dist_path)
    plt.close()
    print(f" -> Saved: {dist_path}")

    # 2. Gender Demographic Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gender_counts = df["Gender"].value_counts()
    axes[0].pie(
        gender_counts, 
        labels=gender_counts.index, 
        autopct="%1.1f%%", 
        colors=["#ff70a6", "#70d6ff"], 
        startangle=140, 
        explode=(0.04, 0),
        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
    )
    axes[0].set_title("Customer Gender Proportion", fontsize=13, weight="bold")

    sns.boxplot(x="Gender", y="Spending_Score", data=df, ax=axes[1], palette=["#ff70a6", "#70d6ff"])
    axes[1].set_title("Spending Score by Gender", fontsize=13, weight="bold")
    axes[1].set_ylabel("Spending Score (1-100)", fontsize=11)

    plt.tight_layout()
    gender_path = os.path.join(VIS_DIR, "02_gender_analysis.png")
    plt.savefig(gender_path)
    plt.close()
    print(f" -> Saved: {gender_path}")

    # 3. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_df = df[["Age", "Annual_Income_k$", "Spending_Score"]].corr()
    sns.heatmap(corr_df, annot=True, cmap="mako", fmt=".2f", vmin=-1, vmax=1, linewidths=1)
    plt.title("Correlation Matrix of Customer Attributes", fontsize=14, weight="bold", pad=12)
    plt.tight_layout()
    corr_path = os.path.join(VIS_DIR, "03_correlation_matrix.png")
    plt.savefig(corr_path)
    plt.close()
    print(f" -> Saved: {corr_path}")


def find_optimal_clusters(X):
    print("\n[3/7] Computing Optimal Cluster Count (Elbow Method & Silhouette Analysis)...")
    k_range = range(2, 11)
    wcss = []
    silhouette_scores = []
    
    # For elbow we include k=1
    wcss_full = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(X)
        wcss_full.append(km.inertia_)
        if k >= 2:
            silhouette_scores.append(silhouette_score(X, km.labels_))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Elbow curve
    ax1.plot(range(1, 11), wcss_full, marker="o", color="#3a86ff", linewidth=2.5, markersize=8)
    ax1.axvline(x=5, color="#e63946", linestyle="--", linewidth=1.8, label="Elbow Point (k=5)")
    ax1.set_title("The Elbow Method (WCSS vs k)", fontsize=13, weight="bold")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=11)
    ax1.legend()
    
    # Silhouette curve
    ax2.plot(k_range, silhouette_scores, marker="s", color="#38b000", linewidth=2.5, markersize=8)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    ax2.axvline(x=optimal_k, color="#e63946", linestyle="--", linewidth=1.8, label=f"Max Silhouette (k={optimal_k})")
    ax2.set_title("Silhouette Score vs k", fontsize=13, weight="bold")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.legend()
    
    plt.tight_layout()
    opt_path = os.path.join(VIS_DIR, "04_elbow_method_silhouette.png")
    plt.savefig(opt_path)
    plt.close()
    print(f" -> Saved: {opt_path}")
    print(f" -> Optimal cluster count selected: k=5 (WCSS elbow at 5, Silhouette: {silhouette_scores[5-2]:.3f})")
    
    return 5


def train_kmeans_model(df, optimal_k=5):
    print("\n[4/7] Training K-Means Clustering Model (k=5)...")
    X = df[["Annual_Income_k$", "Spending_Score"]].values
    
    kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42, n_init=15)
    labels = kmeans.fit_predict(X)
    df["Cluster"] = labels
    
    # Map cluster numbers to meaningful business personas based on their centroid characteristics
    centroids = kmeans.cluster_centers_
    
    persona_map = {}
    for i, (inc, spend) in enumerate(centroids):
        if inc > 70 and spend > 60:
            persona_map[i] = "Champions (High Income, High Spending)"
        elif inc < 45 and spend > 60:
            persona_map[i] = "Spenders / Enthusiasts (Low Income, High Spending)"
        elif inc > 70 and spend < 40:
            persona_map[i] = "Savers / High Potential (High Income, Low Spending)"
        elif inc < 45 and spend < 40:
            persona_map[i] = "Budget / Conservative (Low Income, Low Spending)"
        else:
            persona_map[i] = "Mainstream / Moderate (Mid Income, Mid Spending)"
            
    df["Persona"] = df["Cluster"].map(persona_map)
    
    # Silhouette & Cluster Quality Metrics
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    print(f"Clustering Performance Metrics (k={optimal_k}):")
    print(f" • Silhouette Score:          {sil:.4f} (> 0.5 indicates strong cluster separation)")
    print(f" • Calinski-Harabasz Index:   {ch:.2f} (higher is better)")
    print(f" • Davies-Bouldin Index:      {db:.4f} (lower is better)")
    
    # Plot K-Means Clusters
    plt.figure(figsize=(12, 8))
    colors = ["#e63946", "#1d3557", "#2a9d8f", "#e76f51", "#9d4edd"]
    
    for cluster_id, color in zip(range(optimal_k), colors):
        subset = df[df["Cluster"] == cluster_id]
        p_name = persona_map[cluster_id]
        plt.scatter(
            subset["Annual_Income_k$"], 
            subset["Spending_Score"], 
            s=80, 
            c=color, 
            label=f"Cluster {cluster_id}: {p_name} (n={len(subset)})",
            edgecolors="black",
            linewidth=0.6,
            alpha=0.85
        )
    
    # Plot Centroids
    plt.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        s=300, 
        c="gold", 
        marker="*", 
        edgecolors="black", 
        linewidth=1.5,
        label="Cluster Centroids"
    )
    
    plt.title("Customer Segmentation via K-Means (Annual Income vs Spending Score)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Annual Income (k$)", fontsize=12)
    plt.ylabel("Spending Score (1-100)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    
    kmeans_path = os.path.join(VIS_DIR, "05_kmeans_clusters_2d.png")
    plt.savefig(kmeans_path)
    plt.close()
    print(f" -> Saved: {kmeans_path}")
    
    return kmeans, persona_map


def hierarchical_clustering(df):
    print("\n[5/7] Performing Hierarchical Agglomerative Clustering & Dendrogram...")
    X = df[["Annual_Income_k$", "Spending_Score"]].values
    
    # Generate Linkage Matrix (Ward's method)
    linked = linkage(X, method="ward")
    
    plt.figure(figsize=(14, 6))
    dendrogram(
        linked,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=45.,
        leaf_font_size=10.,
        show_contracted=True
    )
    plt.axhline(y=150, color='r', linestyle='--', label='Cut Threshold (5 Clusters)')
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=14, weight="bold")
    plt.xlabel("Cluster Size / Data Points", fontsize=11)
    plt.ylabel("Euclidean Distance", fontsize=11)
    plt.legend()
    plt.tight_layout()
    
    dendro_path = os.path.join(VIS_DIR, "07_hierarchical_dendrogram.png")
    plt.savefig(dendro_path)
    plt.close()
    print(f" -> Saved: {dendro_path}")


def generate_cluster_insights_and_export(df, persona_map):
    print("\n[6/7] Generating Cluster Persona Profiles & Business Recommendations...")
    
    summary = df.groupby(["Cluster", "Persona"]).agg(
        Count=("CustomerID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Annual_Income_k$", "mean"),
        Avg_Spending=("Spending_Score", "mean"),
        Male_Pct=("Gender", lambda g: (g == "Male").mean() * 100),
        Female_Pct=("Gender", lambda g: (g == "Female").mean() * 100)
    ).reset_index()
    
    print("\n" + "="*80)
    print("CUSTOMER SEGMENTATION EXECUTIVE SUMMARY")
    print("="*80)
    for idx, row in summary.iterrows():
        print(f"\n[Segment {row['Cluster']}] {row['Persona']}")
        print(f" • Population:       {row['Count']} customers ({row['Count']/len(df)*100:.1f}%)")
        print(f" • Average Age:      {row['Avg_Age']:.1f} years")
        print(f" • Avg Annual Income: ${row['Avg_Income']:.1f}k")
        print(f" • Avg Spending Score: {row['Avg_Spending']:.1f}/100")
        print(f" • Gender Ratio:     {row['Female_Pct']:.1f}% Female | {row['Male_Pct']:.1f}% Male")
    
    # Visualization: Radar / Profile Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["Avg_Age", "Avg_Income", "Avg_Spending"]
    titles = ["Average Age by Segment", "Average Income ($k) by Segment", "Average Spending Score by Segment"]
    palettes = ["Blues_d", "Greens_d", "Reds_d"]
    
    for ax, metric, title, pal in zip(axes, metrics, titles, palettes):
        sns.barplot(data=summary, x="Cluster", y=metric, ax=ax, palette=pal)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Cluster ID", fontsize=11)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=11)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha='center', va='center', color='white', weight='bold', fontsize=11)
            
    plt.tight_layout()
    prof_path = os.path.join(VIS_DIR, "06_customer_segment_profiles.png")
    plt.savefig(prof_path)
    plt.close()
    print(f"\n -> Saved: {prof_path}")
    
    # Save enriched CSV
    output_csv = os.path.join(DATA_DIR, "segmented_customers.csv")
    df.to_csv(output_csv, index=False)
    print(f" -> Enriched Segmented Dataset saved: {output_csv}")
    
    # Save JSON summary for web dashboard
    summary_dict = summary.to_dict(orient="records")
    json_path = os.path.join(BASE_DIR, "dashboard", "cluster_data.json")
    with open(json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f" -> Exported Cluster Metadata JSON: {json_path}")
    

def main():
    print("="*80)
    print("THIRANEX INTERNSHIP: CUSTOMER SEGMENTATION ML PIPELINE")
    print("="*80)
    df = load_and_inspect_data()
    perform_eda(df)
    
    X = df[["Annual_Income_k$", "Spending_Score"]].values
    optimal_k = find_optimal_clusters(X)
    kmeans_model, persona_map = train_kmeans_model(df, optimal_k)
    hierarchical_clustering(df)
    generate_cluster_insights_and_export(df, persona_map)
    
    print("\n" + "="*80)
    print(" Pipeline Execution Complete! All artifacts generated successfully.")
    print("="*80)


if __name__ == "__main__":
    main()
