#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 07:01:53 2025

@author: stevezheng
"""

import pandas as pd
import numpy as np

df = pd.read_csv("SpotifyFeatures.csv")
df['mood_score'] = (df['energy'] + df['valence']) / 2
df['dance_energy_ratio'] = df['danceability'] / (df['energy'] + 1e-6)
df.head()
features = [
    'danceability','energy','valence','acousticness','speechiness',
    'loudness','tempo','mood_score','dance_energy_ratio'
]

X = df[features].copy()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

inertia = []
Ks = range(2, 10)

for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(Ks, inertia, marker='o')
plt.title("Figure 1. Elbow Curve")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ---- use a smaller sample for clustering & PCA ----
X_clust = X.sample(n=3000, random_state=42)  # you can even use 2000 if needed

# --- K-Means clustering on the sample ---
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_clust)

# --- Silhouette Score ---
silhouette = silhouette_score(X_clust, clusters)
print("Silhouette Score:", silhouette)

# --- PCA for visualization (also on the same sample) ---
pca = PCA(n_components=2)
vals = pca.fit_transform(X_clust)

plt.figure(figsize=(6,4))
plt.scatter(vals[:,0], vals[:,1], c=clusters, cmap='tab10', s=10)
plt.title("Figure 2. PCA Visualization of Clusters (Sampled Data)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)

# Metrics
print("R2:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

# Feature importance plot
importances = tree.feature_importances_

plt.figure(figsize=(6,4))
plt.barh(features, importances)
plt.title("Figure 3. Feature Importance")
plt.xlabel("Importance")
plt.show()