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
from sklearn.decomposition import PCA, TruncatedSVD
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

# directory setup
output_dir = "embedding_outputs"
os.makedirs(output_dir, exist_ok=True)
dirs = {
    'embeddings': os.path.join(output_dir, 'embeddings'),
    'plots': os.path.join(output_dir, 'plots'),
    'data': os.path.join(output_dir, 'data')
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)
print("Output directories created")
cache_dir = os.path.expanduser("~/gensim_cache")
os.makedirs(cache_dir, exist_ok=True)

########################################
# 1. DATA LOADING & PREPROCESSING
########################################
# load speeches dataset
df = pd.read_csv("./nz_files/nz_filtered.csv")
print("Data loaded")

# clean text 
## using spacy and spacy_cleaner

model_cleanse = spacy.load("en_core_web_sm")

pipeline = Cleaner(
    model_cleanse,
    removers.remove_stopword_token,
    removers.remove_punctuation_token,
    mutators.mutate_lemma_token
)

def sanitize_text(df, speech_column, pipeline):
    speech_list = df[speech_column].to_list()
    speech_list = pipeline.clean(speech_list)
    df[speech_column] = speech_list

df_sample = df.sample(n=5000, random_state=RANDOM_SEED)
    
sanitize_text(df_sample, "text", pipeline)

# Export and save

df_sample.to_csv(os.path.join(dirs['data'], 'NZ_cleaned_sample.csv'), index=False)

########################################
# 2. EMBEDDINGS
########################################
embeddings_dict = {}

texts = df_sample["text"]
tokenized_texts = [text.split() for text in texts]

## A) Word-level embeddings

# Train word2vec

wordlevelmodel = Word2Vec(
    sentences=tokenized_texts,
    vector_size=300,
    window=8,
    min_count=5,
    workers=1,
    seed=RANDOM_SEED # set earlier
)

wordlevelmodel.save(os.path.join(dirs['embeddings'], 'NZ_word2vec_sample.model'))

# aggregate per speech (mean of word vectors)

word_embeddings = []
for tokens in tokenized_texts:
    vecs = [wordlevelmodel.wv[token] for token in tokens if token in wordlevelmodel.wv]
    if vecs:
        speech_vec = np.mean(vecs, axis=0) #np.mean(vecs, axis=0) computes the average for each of the 300 dimensions across all words
    else:
        speech_vec = np.zeros(wordlevelmodel.vector_size)
    word_embeddings.append(speech_vec)

word_embeddings = np.vstack(word_embeddings)

print(word_embeddings)
# Each row corresponds to one speech, each column is one of the 300 vectors

embeddings_dict["word"] = word_embeddings

## B) Sentence-level embeddings
# use sentence-transformers model (MiniLM, all-MiniLM-L6-v2 etc.)
# batch encode speeches
embeddings_dict["sentence"] = sentence_embeddings

## C) Topic-level embeddings
# vectorize text with TF-IDF → run LDA
# represent each speech by topic distribution vector
embeddings_dict["topic"] = lda_embeddings

## D) Context-aware embeddings
# smaller transformer model (distilbert-base-uncased, etc.)
embeddings_dict["context"] = context_embeddings

# save embeddings
for emb_type, emb in embeddings_dict.items():
    np.save(f"outputs/embeddings/{emb_type}_embeddings.npy", emb)



