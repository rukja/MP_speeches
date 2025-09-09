########################################
# 0. SETUP
########################################
# imports
import os
import random
import warnings
#warnings.filterwarnings('ignore')
import scipy
import pickle
import numpy as np
import pandas as pd
import umap
import umap.plot
import seaborn as sns
import matplotlib.pyplot as plt
import bokeh 
from bokeh.plotting import figure, show, output_notebook, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Category10
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
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

cluster_labels = {}

for emb_type, emb in embeddings_dict.items():
    print(f"Starting unsupervised clustering for {emb_type}")
    for method in ["kmeans"]:
        print(f"Using {method}")
        for k in range(2, 16):  # try cluster numbers
            # fit clustering
            if method == "kmeans":
                model = KMeans(n_clusters=k, random_state=RANDOM_SEED)
            else:
                model = AgglomerativeClustering(n_clusters=k)

            labels = model.fit_predict(emb)

            # evaluate
            score = silhouette_score(emb, labels)

            # log result
            results.append([emb_type, method, k, score])

            # save cluster labels
            col_name = f"{emb_type}_{method}_{k}"
            cluster_labels[col_name] = labels


# Re add results 
labels_df = pd.DataFrame(cluster_labels, index=df_sample.index)
results_df = pd.DataFrame(results, columns=["embedding", "method", "k", "silhouette"])
# save clustering results
results_df.to_csv("embedding_outputs/clustering/NZ_sample_silhouette_scores.csv", index=False)

# Visualize

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=results_df,
    x="k",
    y="silhouette",
    hue="embedding",
    style="method",
    marker="o"
)

plt.title("Silhouette Scores by Embedding Type & Clustering Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.legend(title="Embedding / Method", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
plt.savefig("embedding_outputs/plots/NZ_Silhouette.png")

# Select best clustering method

best_clustering_row = results_df.loc[results_df['silhouette'].idxmax()]
best_clustering = "_".join(best_clustering_row[['embedding', 'method', 'k']].astype(str))
print(best_clustering)
best_labels = pd.DataFrame(labels_df[best_clustering], columns = [best_clustering])

df_sample = pd.concat([df_sample, best_labels], axis=1)

df_sample.to_csv("embedding_outputs/clustering/NZ_sample_clustered_speeches.csv", index=False)

########################################
# 2. Unsupervised Clustering Visualization
########################################
print(embeddings_dict['topic'][0])
print(embeddings_dict['topic'][0].sum())
topic_scaled = StandardScaler().fit_transform(embeddings_dict['topic'])

labels = labels_df[best_clustering].to_numpy()  # 1D array of ints
mapper = umap.UMAP(random_state=RANDOM_SEED).fit(topic_scaled)

embedding_2d = mapper.embedding_
k = len(np.unique(labels))
cmap = plt.cm.get_cmap("tab10", k)
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    embedding_2d[:,0],
    embedding_2d[:,1],
    c=labels,
    cmap=cmap,
    s=10
)
plt.title("UMAP projection of topic embeddings")
plt.show()
plt.savefig("embedding_outputs/plots/NZ_umap_topic.png")

# interactive with party

df_plot = pd.DataFrame({
    "UMAP1": embedding_2d[:,0],
    "UMAP2": embedding_2d[:,1],
    "Cluster": labels,
    "Party": df_sample["party"]  
})
df_plot["Cluster"] = df_plot["Cluster"].astype(str)
source = ColumnDataSource(df_plot)



clusters = df_plot["Cluster"].unique()
pal = Category10[len(clusters)]
color_mapper = CategoricalColorMapper(factors=clusters, palette=pal)

# Create plot
p = figure(title="UMAP projection of topic embeddings",
           width=800, height=600,
           tools="pan,wheel_zoom,box_zoom,reset,save")

# Add circles
p.circle(
    x="UMAP1",
    y="UMAP2",
    source=source,
    color={"field": "Cluster", "transform": color_mapper},
    size=6,
    alpha=0.8
)

# Add hover tool
hover = HoverTool(tooltips=[
    ("Party", "@Party"),
    ("Cluster", "@Cluster")
])
p.add_tools(hover)

show(p)

output_file("embedding_outputs/plots/NZ_interactive_party_UMAP_topic.html")
save(p)

# with party
plt.figure(figsize=(8,6))
scatter = sns.scatterplot(
    x = df_plot["UMAP1"],
    y = df_plot["UMAP2"],
    hue=df_plot["Party"],
    style=df_plot["Cluster"],
    palette = "Accent"
)
plt.title("UMAP projection of topic embeddings with party", fontsize=14)


plt.legend(fontsize=8, title_fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()
plt.savefig("embedding_outputs/plots/NZ_umap_topic_party.png")

