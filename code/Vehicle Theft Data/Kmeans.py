import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv('data/car data.csv')

# Step 1: Feature Engineering
# Add Age of the car
data['Age'] = 2024 - data['Year']

# Step 2: Data Preprocessing
# Convert categorical variables to numeric using one-hot encoding
data = pd.get_dummies(data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Select numeric features
numeric_features = ['Present_Price', 'Kms_Driven', 'Year', 'Age']  # You can add more features as needed
X = data[numeric_features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Hyperparameter Tuning (Finding Optimal k using Elbow Method)
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # Elbow method
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))  # Silhouette score

# Plot the elbow method
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--', color='green')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Based on the plots, choose the optimal number of clusters
optimal_k = 4  # Change this based on your plots

# Step 4: Apply K-Means Clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 5: Cluster Interpretation
# Visualize the clusters in 2D (using the first two features)
plt.figure(figsize=(8, 6))
plt.scatter(data['Present_Price'], data['Kms_Driven'], c=data['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Cars (Present Price vs Kms Driven)')
plt.xlabel('Present Price')
plt.ylabel('Kms Driven')
plt.colorbar(label='Cluster')
plt.show()

# Optionally: Show the cluster centers (in the original scale)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=numeric_features)
print("Cluster Centers (in original scale):")
print(cluster_centers_df)