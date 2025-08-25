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

K = range(2, 10)
fits = []
score = []


for k in K:
    
    print("k")
    model = KMeans(n_clusters = k, random_state = 19, n_init='auto')
    
    # append the model to fits
    big_data["cluster"] = model.fit_predict(X)
    print("Model fit completed")
    fits.append(model)
    
    # Append the silhouette score to scores
    score.append(silhouette_score(X, model.labels_, metric='euclidean'))
    print("Silhoutte scores completed")

sns.lineplot(x = K, y = score)