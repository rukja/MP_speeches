# General Libraries Needed
import glob, csv
import pandas as pd
from collections import defaultdict, Counter
from lxml import etree
import qsck
import re
from sklearn.metrics import silhouette_score

# Functions for Unsupervised Clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Libraries for Graphing
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


big_data = pd.read_csv("./nz_files/nz_filtered.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)       # remove numbers
    text = re.sub(r'\W+', ' ', text)      # remove punctuation
    text = re.sub(r'\s+', ' ', text)      # remove extra spaces
    return text.strip()

big_data["clean_text"] = big_data["text"].apply(clean_text)

# Vectorize

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(big_data["clean_text"])
print("TF-IDF shape:", X.shape)

# Cluster

from sklearn.cluster import KMeans

k = 5  
kmeans = KMeans(n_clusters=k, random_state=19, n_init='auto')
big_data["cluster"] = kmeans.fit_predict(X)

print(big_data["cluster"].value_counts())

party_counts = big_data.groupby(["cluster", "party"]).size().reset_index(name="count")
print(party_counts.head())

plt.figure(figsize=(12,6))
sns.barplot(data=party_counts, x="cluster", y="count", hue="party")

plt.title("Distribution of Parties within Clusters")
plt.xlabel("Cluster")
plt.ylabel("Number of Speeches")
plt.legend(title="Party", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Create a cross-tab (clusters x parties)
ct = pd.crosstab(big_data["cluster"], big_data["party"])

# Plot stacked bar chart
ct.plot(kind="bar", stacked=True, figsize=(12,6), colormap="tab20")

plt.title("Party Distribution within Clusters (Stacked)")
plt.xlabel("Cluster")
plt.ylabel("Number of Speeches")
plt.legend(title="Party", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

from sklearn.decomposition import TruncatedSVD

# Reduce to 2 components for plotting
svd = TruncatedSVD(n_components=2, random_state=19)
X_reduced = svd.fit_transform(X)

big_data["pc1"] = X_reduced[:,0]
big_data["pc2"] = X_reduced[:,1]

# Scatterplot of speeches in PCA space, colored by cluster
plt.figure(figsize=(10,7))
sns.scatterplot(data=big_data.sample(5000, random_state=42),  # sample for readability
                x="pc1", y="pc2", hue="party", palette="tab10", alpha=0.6)

plt.title("Clusters of MP Speeches (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
