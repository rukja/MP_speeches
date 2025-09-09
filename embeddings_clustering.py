########################################
# 0. SETUP
########################################
# imports
import os
import random
import warnings
#warnings.filterwarnings('ignore')
import nltk
import scipy
import spacy
import pickle
import numpy as np
import pandas as pd
import gensim
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
from spacy_cleaner import Cleaner
from spacy_cleaner.processing import removers, replacers, mutators
from gensim import corpora
from gensim.models import Word2Vec
from pprint import pprint

# seeds for reproducibility
RANDOM_SEED = 19
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Read in embeddings
embeddings_dict = {}

hybrid_reduced_embeddings = np.load("embedding_outputs/embeddings/hybrid_embeddings.npy")
sentence_embeddings = np.load("embedding_outputs/embeddings/sentence_embeddings.npy")
topic_embeddings = np.load("embedding_outputs/embeddings/topic_embeddings.npy")
word_embeddings = np.load("embedding_outputs/embeddings/word_embeddings.npy")

embeddings_dict['hybrid'] = hybrid_reduced_embeddings
embeddings_dict['sentence'] = sentence_embeddings
embeddings_dict['topic'] = topic_embeddings
embeddings_dict['word'] = word_embeddings

np.save(f"embedding_outputs/embeddings/embeddings_dict.npy", embeddings_dict)

# Read in sampled df
df_sample = pd.read_csv("embedding_outputs/data/NZ_cleaned_sample.csv")

# Create clustering folder
clustering_output = "embeddings_outputs/clustering"
os.makedirs(clustering_output, exist_ok=True)

########################################
# 1. Unsupervised Clustering
########################################
results = []
print("Starting unsupervised clustering")

for emb_type, emb in embeddings_dict.items():
    print(f"Starting unsupervised clustering for {emb_type}")
    for method in ["kmeans", "agglomerative", "spectral"]:
        print(f"Using {method}")
        for k in range(2, 16):  # try cluster numbers
            # fit clustering
            if method == "kmeans":
                model = KMeans(n_clusters=k, random_state=RANDOM_SEED)
            elif method == "agglomerative":
                model = AgglomerativeClustering(n_clusters=k)
            else:
                model = SpectralClustering(n_clusters=k, random_state=RANDOM_SEED)

            labels = model.fit_predict(emb)

            # evaluate
            score = silhouette_score(emb, labels)

            # log result
            results.append([emb_type, method, k, score])

            # save cluster labels
            df_sample[f"{emb_type}_{method}_{k}"] = labels

# save clustering results
pd.DataFrame(results, columns=["embedding", "method", "k", "silhouette"]) \
  .to_csv("embedding_outputs/clustering/NZ_sample_silhouette_scores.csv", index=False)

df_sample.to_csv("embedding_outputs/clustering/NZ_sample_clustered_speeches.csv", index=False)