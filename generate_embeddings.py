########################################
# 0. SETUP
########################################
# imports
import os
import random
import warnings
import nltk
import scipy
import spacy
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
from spacy_cleaner import processing, Cleaner

# seeds for reproducibility
RANDOM_SEED = 19
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# directory setup
output_dir = "embedding_outputs"
os.makedirs(output_dir, exist_ok=True)
dirs = {
    'embeddings': os.path.join(output_dir, 'embeddings'),
    'plots': os.path.join(output_dir, 'plots'),
    'data': os.path.join(output_dir, 'data')
            }

########################################
# 1. DATA LOADING & PREPROCESSING
########################################
# load speeches dataset
df = pd.read_csv("./nz_files/nz_filtered.csv")

# clean text 
## using spacy and spacy_cleaner

model_cleanse = spacy.load("en_core_web_sm")
pipeline = spacy_cleaner.Pipeline(
    model = model_cleanse,
    removers.remove_stopward_token,
    replacers.replace_punctuation_token,
    mutators.mutate_lemma_token,
)
def sanitize_text(speech_column):


