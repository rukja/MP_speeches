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
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
batch_size = 32 if len(texts) > 1000 else 64
sentence_embeddings = sentence_model.encode(
        texts.tolist(), 
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
)

# batch encode speeches
embeddings_dict["sentence"] = sentence_embeddings

## C) Topic-level embeddings
# vectorize text with TF-IDF → run LDA
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=20,
    stop_words="english"
)

tfidf_matrix = tfidf_vectorizer.fit_transform(texts.tolist())
# represent each speech by topic distribution vector
n_topics = 15
lda_model = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=RANDOM_SEED, # set earlier
    learning_method="batch"
)
lda_topics = lda_model.fit_transform(tfidf_matrix)


# exploration of topics 
def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx}: {' '.join(top_words)}")

print_top_words(lda_model, tfidf_vectorizer.get_feature_names_out())

embeddings_dict["topic"] = lda_topics

## D) Hybrid embeddings

normalized_embeddings = {}
for name, embeddings in embeddings_dict.items():
    scaler = StandardScaler()
    normalized_embeddings[name] = scaler.fit_transform(embeddings)

hybrid = np.hstack([normalized_embeddings[name] for name in embeddings_dict]) # Works because we have the same number of speeches

n_components = min(100, hybrid.shape[1] // 2)
pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
hybrid_reduced = pca.fit_transform(hybrid)
embeddings_dict['hybrid'] = hybrid_reduced

# save embeddings
for emb_type, emb in embeddings_dict.items():
    np.save(f"embedding_outputs/embeddings/{emb_type}_embeddings.npy", emb)



