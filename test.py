# CPU-Optimized Contextual Text Clustering Analysis with Output Saving
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
import os
import json
import pickle
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# Context-aware embeddings (CPU optimized)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

plt.style.use('ggplot')

class ContextAwareTextClustererWithOutputs:
    def __init__(self, data, text_column, label_column=None, sample_size=None, device='cpu', output_dir='clustering_outputs'):
        """
        Context-aware text clustering with comprehensive output saving
        
        Parameters:
        - output_dir: Directory to save all outputs (created if doesn't exist)
        """
        self.original_data = data.copy()
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        self.dirs = {
            'plots': os.path.join(self.output_dir, 'plots'),
            'data': os.path.join(self.output_dir, 'data'),
            'models': os.path.join(self.output_dir, 'models'),
            'reports': os.path.join(self.output_dir, 'reports'),
            'embeddings': os.path.join(self.output_dir, 'embeddings')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Sample data if specified
        if sample_size and len(data) > sample_size:
            self.data = data.sample(n=sample_size, random_state=42).copy()
            print(f"Sampled {sample_size} rows from {len(data)} total rows for CPU efficiency")
        else:
            self.data = data.copy()
            
        self.text_column = text_column
        self.label_column = label_column
        self.embeddings = {}
        self.clustering_results = {}
        self.models = {}
        
        # Force CPU usage
        if torch.cuda.is_available():
            print("CUDA detected but forcing CPU usage as requested")
        self.device = 'cpu'
        
        # Initialize log
        self.log = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': self.data.shape,
            'text_column': text_column,
            'label_column': label_column,
            'sample_size': sample_size,
            'steps_completed': []
        }
        
    def save_log(self):
        """Save processing log"""
        with open(os.path.join(self.dirs['reports'], 'processing_log.json'), 'w') as f:
            json.dump(self.log, f, indent=2)
    
    def enhanced_text_cleaning(self, text, preserve_structure=True):
        """Clean text while preserving sentence structure for embeddings"""
        text = str(text)
        
        if preserve_structure:
            # Minimal cleaning to preserve context
            text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
            text = re.sub(r'\S+@\S+', '[EMAIL]', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        else:
            # More aggressive cleaning
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    
    def create_sentence_embeddings(self, model_names=['all-MiniLM-L6-v2']):
        """Create sentence embeddings and save them"""
        print("Creating sentence embeddings...")
        self.data['clean_text'] = self.data[self.text_column].apply(
            lambda x: self.enhanced_text_cleaning(x, preserve_structure=True)
        )
        
        # Truncate very long texts
        max_length = 500
        self.data['clean_text'] = self.data['clean_text'].apply(
            lambda x: x[:max_length] if len(x) > max_length else x
        )
        
        texts = self.data['clean_text'].tolist()
        
        embedding_info = {
            'model_names': model_names,
            'text_count': len(texts),
            'max_length': max_length,
            'embedding_shapes': {}
        }
        
        for model_name in model_names:
            print(f"Loading sentence transformer: {model_name}")
            try:
                model = SentenceTransformer(model_name, device=self.device)
                model.eval()
                
                print(f"Encoding {len(texts)} texts with {model_name}...")
                
                batch_size = 32 if len(texts) > 1000 else 64
                embeddings = model.encode(
                    texts, 
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                emb_name = f'sentence_{model_name.replace("-", "_")}'
                self.embeddings[emb_name] = embeddings
                self.models[model_name] = model
                embedding_info['embedding_shapes'][emb_name] = embeddings.shape
                
                # Save embeddings
                np.save(os.path.join(self.dirs['embeddings'], f'{emb_name}.npy'), embeddings)
                print(f"  Saved embeddings to {emb_name}.npy")
                
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
                continue
        
        # Save embedding info
        with open(os.path.join(self.dirs['reports'], 'embedding_info.json'), 'w') as f:
            json.dump(embedding_info, f, indent=2)
        
        self.log['steps_completed'].append('sentence_embeddings')
        self.save_log()
    
    def create_doc2vec_like_embeddings(self):
        """Create and save Doc2Vec-like embeddings"""
        print("Creating Doc2Vec-like embeddings...")
        
        clean_texts = self.data[self.text_column].apply(
            lambda x: self.enhanced_text_cleaning(x, preserve_structure=False)
        )
        
        # Create TF-IDF weighted embeddings
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', 
                              min_df=0.01, max_df=0.8, ngram_range=(1,1))
        tfidf_matrix = tfidf.fit_transform(clean_texts)
        feature_names = tfidf.get_feature_names_out()
        
        vocab_size = len(feature_names)
        embedding_dim = 100
        
        np.random.seed(42)
        word_embeddings = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        
        doc_embeddings = []
        tfidf_dense = tfidf_matrix.toarray()
        
        for doc_tfidf in tfidf_dense:
            nonzero_indices = np.nonzero(doc_tfidf)[0]
            if len(nonzero_indices) == 0:
                doc_embedding = np.zeros(embedding_dim)
            else:
                weights = doc_tfidf[nonzero_indices]
                word_vecs = word_embeddings[nonzero_indices]
                doc_embedding = np.average(word_vecs, axis=0, weights=weights)
            doc_embeddings.append(doc_embedding)
        
        doc_embeddings = np.array(doc_embeddings)
        self.embeddings['doc2vec_like'] = doc_embeddings
        
        # Save embeddings and model components
        np.save(os.path.join(self.dirs['embeddings'], 'doc2vec_like.npy'), doc_embeddings)
        np.save(os.path.join(self.dirs['embeddings'], 'word_embeddings.npy'), word_embeddings)
        
        # Save TF-IDF model
        with open(os.path.join(self.dirs['models'], 'tfidf_model.pkl'), 'wb') as f:
            pickle.dump(tfidf, f)
        
        print(f"  Created and saved Doc2Vec-like embeddings shape: {doc_embeddings.shape}")
        
        self.log['steps_completed'].append('doc2vec_like_embeddings')
        self.save_log()
    
    def create_topic_embeddings(self, n_topics=20):
        """Create and save topic-based embeddings"""
        print(f"Creating topic embeddings with {n_topics} topics...")
        
        clean_texts = self.data[self.text_column].apply(
            lambda x: self.enhanced_text_cleaning(x, preserve_structure=False)
        )
        
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(max_features=3000, stop_words='english', 
                                   min_df=0.01, max_df=0.8)
        doc_term_matrix = vectorizer.fit_transform(clean_texts)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
        topic_distributions = lda.fit_transform(doc_term_matrix)
        
        self.embeddings['topic_lda'] = topic_distributions
        self.models['lda'] = lda
        self.models['lda_vectorizer'] = vectorizer
        
        # Save topic embeddings and model
        np.save(os.path.join(self.dirs['embeddings'], 'topic_lda.npy'), topic_distributions)
        
        with open(os.path.join(self.dirs['models'], 'lda_model.pkl'), 'wb') as f:
            pickle.dump(lda, f)
        with open(os.path.join(self.dirs['models'], 'lda_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        
        # Save topic analysis
        feature_names = vectorizer.get_feature_names_out()
        topics_report = []
        
        print("  Top words per topic:")
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topic_info = {
                'topic_id': topic_idx,
                'top_words': top_words,
                'word_weights': [float(topic[i]) for i in topic.argsort()[-10:][::-1]]
            }
            topics_report.append(topic_info)
            print(f"    Topic {topic_idx}: {', '.join(top_words[:5])}")
        
        # Save topics report
        with open(os.path.join(self.dirs['reports'], 'topics_analysis.json'), 'w') as f:
            json.dump(topics_report, f, indent=2)
        
        print(f"  Created topic embeddings shape: {topic_distributions.shape}")
        
        self.log['steps_completed'].append('topic_embeddings')
        self.save_log()
    
    def create_hybrid_embeddings(self):
        """Create and save hybrid embeddings"""
        available_embeddings = list(self.embeddings.keys())
        
        if len(available_embeddings) < 2:
            print("Need at least 2 embedding types to create hybrid embeddings")
            return
        
        print("Creating hybrid embeddings...")
        
        # Normalize all embeddings
        normalized_embeddings = {}
        for name, embeddings in self.embeddings.items():
            scaler = StandardScaler()
            normalized_embeddings[name] = scaler.fit_transform(embeddings)
        
        # Concatenate embeddings
        hybrid = np.hstack([normalized_embeddings[name] for name in available_embeddings])
        
        # Apply PCA
        n_components = min(100, hybrid.shape[1] // 2)
        pca = PCA(n_components=n_components, random_state=42)
        hybrid_reduced = pca.fit_transform(hybrid)
        
        self.embeddings['hybrid'] = hybrid_reduced
        
        # Save hybrid embeddings and PCA model
        np.save(os.path.join(self.dirs['embeddings'], 'hybrid.npy'), hybrid_reduced)
        
        with open(os.path.join(self.dirs['models'], 'pca_model.pkl'), 'wb') as f:
            pickle.dump(pca, f)
        
        # Save PCA analysis
        pca_info = {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_explained_variance': float(pca.explained_variance_ratio_.sum()),
            'component_embeddings': available_embeddings
        }
        
        with open(os.path.join(self.dirs['reports'], 'pca_analysis.json'), 'w') as f:
            json.dump(pca_info, f, indent=2)
        
        print(f"  Created hybrid embeddings shape: {hybrid_reduced.shape}")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        self.log['steps_completed'].append('hybrid_embeddings')
        self.save_log()
    
    def semantic_similarity_analysis(self, embedding_type=None, n_samples=100):
        """Analyze and save semantic similarity patterns"""
        if not self.embeddings:
            print("No embeddings found. Create embeddings first.")
            return
        
        if embedding_type is None:
            embedding_type = list(self.embeddings.keys())[0]
        
        if embedding_type not in self.embeddings:
            print(f"Embedding type {embedding_type} not found")
            return
        
        print(f"Analyzing semantic similarity with {embedding_type}...")
        embeddings = self.embeddings[embedding_type]
        
        # Sample for efficiency
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            sample_embeddings = embeddings[indices]
            sample_data = self.data.iloc[indices]
        else:
            sample_embeddings = embeddings
            sample_data = self.data
            indices = range(len(embeddings))
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(sample_embeddings)
        
        # Save similarity matrix
        np.save(os.path.join(self.dirs['data'], f'similarity_matrix_{embedding_type}.npy'), similarity_matrix)
        
        # Plot and save similarity heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='viridis', square=True)
        plt.title(f'Semantic Similarity Matrix ({embedding_type})')
        plt.savefig(os.path.join(self.dirs['plots'], f'similarity_heatmap_{embedding_type}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find and save most similar pairs
        similarity_matrix_upper = np.triu(similarity_matrix, k=1)
        top_pairs_indices = np.unravel_index(
            np.argsort(similarity_matrix_upper.ravel())[-10:], 
            similarity_matrix_upper.shape
        )
        
        similar_pairs = []
        print("\nMost similar document pairs:")
        for i, (idx1, idx2) in enumerate(zip(top_pairs_indices[0], top_pairs_indices[1])):
            if similarity_matrix_upper[idx1, idx2] > 0:
                sim_score = float(similarity_matrix_upper[idx1, idx2])
                text1 = str(sample_data.iloc[idx1][self.text_column])
                text2 = str(sample_data.iloc[idx2][self.text_column])
                
                pair_info = {
                    'pair_id': i + 1,
                    'similarity_score': sim_score,
                    'text1': text1,
                    'text2': text2,
                    'text1_preview': text1[:200] + "..." if len(text1) > 200 else text1,
                    'text2_preview': text2[:200] + "..." if len(text2) > 200 else text2
                }
                similar_pairs.append(pair_info)
                
                print(f"\nPair {i+1} (Similarity: {sim_score:.3f}):")
                print(f"Text 1: {pair_info['text1_preview']}")
                print(f"Text 2: {pair_info['text2_preview']}")
        
        # Save similar pairs analysis
        with open(os.path.join(self.dirs['reports'], f'similar_pairs_{embedding_type}.json'), 'w') as f:
            json.dump(similar_pairs, f, indent=2)
        
        self.log['steps_completed'].append(f'similarity_analysis_{embedding_type}')
        self.save_log()
    
    def perform_contextual_clustering(self, embedding_types=None, k_range=(2, 10)):
        """Perform clustering and save all results"""
        if embedding_types is None:
            embedding_types = list(self.embeddings.keys())
        
        results = {}
        
        for emb_type in embedding_types:
            if emb_type not in self.embeddings:
                continue
                
            print(f"\nClustering with {emb_type} embeddings...")
            X = self.embeddings[emb_type]
            
            # Find optimal k
            silhouette_scores = []
            k_values = list(range(k_range[0], k_range[1] + 1))
            
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                labels = kmeans.fit_predict(X)
                
                if len(np.unique(labels)) > 1:
                    if len(X) > 3000:
                        sample_indices = np.random.choice(len(X), 3000, replace=False)
                        score = silhouette_score(X[sample_indices], labels[sample_indices])
                    else:
                        score = silhouette_score(X, labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(0)
            
            optimal_k = k_values[np.argmax(silhouette_scores)]
            best_score = max(silhouette_scores)
            
            print(f"  Optimal k: {optimal_k} (silhouette: {best_score:.3f})")
            
            # Final clustering
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            final_labels = final_kmeans.fit_predict(X)
            
            results[emb_type] = {
                'k': optimal_k,
                'labels': final_labels.tolist(),
                'silhouette_score': best_score,
                'silhouette_scores': silhouette_scores,
                'k_values': k_values,
                'cluster_centers': final_kmeans.cluster_centers_.tolist()
            }
            
            # Save clustering results
            np.save(os.path.join(self.dirs['data'], f'cluster_labels_{emb_type}.npy'), final_labels)
            
            # Save clustering model
            with open(os.path.join(self.dirs['models'], f'kmeans_model_{emb_type}.pkl'), 'wb') as f:
                pickle.dump(final_kmeans, f)
        
        self.clustering_results = results
        
        # Save all clustering results
        with open(os.path.join(self.dirs['reports'], 'clustering_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create and save comparison plot
        plt.figure(figsize=(15, 10))
        n_plots = len(results)
        cols = 2
        rows = (n_plots + cols - 1) // cols
        
        for i, (emb_type, result) in enumerate(results.items()):
            plt.subplot(rows, cols, i + 1)
            plt.plot(result['k_values'], result['silhouette_scores'], 'o-', linewidth=2, markersize=8)
            plt.axvline(x=result['k'], color='red', linestyle='--', alpha=0.7, linewidth=2)
            plt.title(f'{emb_type}\nOptimal k: {result["k"]} (score: {result["silhouette_score"]:.3f})', fontsize=12)
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['plots'], 'clustering_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        self.log['steps_completed'].append('contextual_clustering')
        self.save_log()
        
        return results
    
    def analyze_semantic_clusters(self, embedding_type=None, top_examples=3):
        """Analyze clusters and save detailed reports"""
        if not self.clustering_results:
            print("No clustering results found. Run clustering first.")
            return
        
        if embedding_type is None:
            embedding_type = list(self.clustering_results.keys())[0]
        
        labels = np.array(self.clustering_results[embedding_type]['labels'])
        self.data[f'cluster_{embedding_type}'] = labels
        
        print(f"\nSemantic Cluster Analysis - {embedding_type}")
        print("=" * 60)
        
        cluster_analysis = {
            'embedding_type': embedding_type,
            'total_clusters': len(np.unique(labels[labels >= 0])),
            'clusters': []
        }
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
                
            cluster_mask = labels == cluster_id
            cluster_texts = self.data[cluster_mask][self.text_column]
            cluster_size = len(cluster_texts)
            
            print(f"\nCluster {cluster_id} (Size: {cluster_size})")
            print("-" * 40)
            
            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': cluster_size,
                'representative_texts': []
            }
            
            # Party distribution
            if self.label_column and self.label_column in self.data.columns:
                party_dist = self.data[cluster_mask][self.label_column].value_counts()
                cluster_info['party_distribution'] = dict(party_dist)
                print(f"Party distribution: {dict(party_dist)}")
            
            # Find representative examples
            cluster_embeddings = self.embeddings[embedding_type][cluster_mask]
            cluster_center = cluster_embeddings.mean(axis=0)
            
            distances = np.array([
                cosine(cluster_center, emb) 
                for emb in cluster_embeddings
            ])
            closest_indices = distances.argsort()[:top_examples]
            
            print(f"\nTop {top_examples} representative texts:")
            for i, idx in enumerate(closest_indices):
                text = str(cluster_texts.iloc[idx])
                text_preview = text[:300] + "..." if len(text) > 300 else text
                
                cluster_info['representative_texts'].append({
                    'rank': i + 1,
                    'distance_to_center': float(distances[idx]),
                    'text': text,
                    'preview': text_preview
                })
                
                print(f"  {i+1}. {text_preview}")
                print()
            
            cluster_analysis['clusters'].append(cluster_info)
        
        # Save cluster analysis
        with open(os.path.join(self.dirs['reports'], f'cluster_analysis_{embedding_type}.json'), 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        
        # Save data with cluster labels
        output_data = self.data.copy()
        output_data.to_csv(os.path.join(self.dirs['data'], f'data_with_clusters_{embedding_type}.csv'), index=False)
        
        self.log['steps_completed'].append(f'semantic_analysis_{embedding_type}')
        self.save_log()
        
        return cluster_analysis
    
    def visualize_semantic_clusters(self, embedding_type=None, method='pca'):
        """Create and save cluster visualizations"""
        if not self.clustering_results:
            print("No clustering results found.")
            return
        
        if embedding_type is None:
            embedding_type = list(self.clustering_results.keys())[0]
        
        X = self.embeddings[embedding_type]
        labels = np.array(self.clustering_results[embedding_type]['labels'])
        
        print(f"Creating 2D visualization using {method}...")
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        X_2d = reducer.fit_transform(X)
        
        # Save 2D coordinates
        coords_df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'cluster': labels
        })
        coords_df.to_csv(os.path.join(self.dirs['data'], f'2d_coordinates_{embedding_type}_{method}.csv'), index=False)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot by clusters
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                color = 'black'
                alpha = 0.3
                marker = 'x'
            else:
                color = colors[i]
                alpha = 0.7
                marker = 'o'
            
            mask = labels == label
            axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                          c=[color], alpha=alpha, s=50, 
                          label=f'Cluster {label}', marker=marker)
        
        axes[0].set_title(f'Semantic Clusters - {embedding_type}', fontsize=14)
        axes[0].set_xlabel(f'{method.upper()} Component 1')
        axes[0].set_ylabel(f'{method.upper()} Component 2')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot by original labels
        if self.label_column and self.label_column in self.data.columns:
            unique_parties = self.data[self.label_column].unique()
            party_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_parties)))
            
            for i, party in enumerate(unique_parties):
                mask = self.data[self.label_column] == party
                axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                              c=[party_colors[i]], alpha=0.7, s=50, 
                              label=party)
            
            axes[1].set_title(f'Original Labels - {self.label_column}', fontsize=14)
            axes[1].set_xlabel(f'{method.upper()} Component 1')
            axes[1].set_ylabel(f'{method.upper()} Component 2')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No original labels available', 
                        transform=axes[1].transAxes, ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['plots'], f'cluster_visualization_{embedding_type}_{method}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        self.log['steps_completed'].append(f'visualization_{embedding_type}_{method}')
        self.save_log()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        report = {
            'analysis_summary': {
                'timestamp': datetime.now().isoformat(),
                'data_shape': self.data.shape,
                'embedding_types_created': list(self.embeddings.keys()),
                'clustering_methods_used': list(self.clustering_results.keys()),
                'steps_completed': self.log['steps_completed']
            },
            'best_clustering_method': None,
            'recommendations': []
        }
        
        # Find best clustering method
        if self.clustering_results:
            best_method = max(self.clustering_results.keys(), 
                            key=lambda x: self.clustering_results[x]['silhouette_score'])
            report['best_clustering_method'] = {
                'method': best_method,
                'silhouette_score': self.clustering_results[best_method]['silhouette_score'],
                'optimal_k': self.clustering_results[best_method]['k']
            }
        
        # Add recommendations
        if len(self.embeddings) > 1:
            report['recommendations'].append("Multiple embedding types were created for robust analysis")
        if 'sentence_' in str(self.embeddings.keys()):
            report['recommendations'].append("Sentence embeddings provide the most context-aware clustering")
        if 'hybrid' in self.embeddings:
            report['recommendations'].append("Hybrid embeddings combine strengths of multiple methods")
        
        # Save comprehensive report
        with open(os.path.join(self.dirs['reports'], 'comprehensive_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create summary statistics
        if self.clustering_results:
            stats_df = pd.DataFrame([
                {
                    'embedding_type': emb_type,
                    'optimal_k': results['k'],
                    'silhouette_score': results['silhouette_score'],
                    'max_silhouette': max(results['silhouette_scores']),
                    'embedding_dimension': self.embeddings[emb_type].shape[1]
                }
                for emb_type, results in self.clustering_results.items()
            ])
            stats_df.to_csv(os.path.join(self.dirs['reports'], 'clustering_statistics.csv'), index=False)
        
        print(f"\n{'='*60}")
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"All outputs saved to: {self.output_dir}")
        print(f"Best clustering method: {report['best_clustering_method']['method'] if report['best_clustering_method'] else 'None'}")
        print(f"Number of embedding types: {len(self.embeddings)}")
        print(f"Steps completed: {len(self.log['steps_completed'])}")
        
        return report
    
    def load_saved_results(self, output_dir=None):
        """Load previously saved results"""
        if output_dir:
            self.output_dir = output_dir
            self.dirs = {
                'plots': os.path.join(self.output_dir, 'plots'),
                'data': os.path.join(self.output_dir, 'data'),
                'models': os.path.join(self.output_dir, 'models'),
                'reports': os.path.join(self.output_dir, 'reports'),
                'embeddings': os.path.join(self.output_dir, 'embeddings')
            }
        
        # Load embeddings
        if os.path.exists(self.dirs['embeddings']):
            embedding_files = [f for f in os.listdir(self.dirs['embeddings']) if f.endswith('.npy')]
            for file in embedding_files:
                emb_name = file.replace('.npy', '')
                self.embeddings[emb_name] = np.load(os.path.join(self.dirs['embeddings'], file))
                print(f"Loaded {emb_name}: {self.embeddings[emb_name].shape}")
        
        # Load clustering results (JSON-serializable version)
        results_file = os.path.join(self.dirs['reports'], 'clustering_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                clustering_summary = json.load(f)
            
            # Reconstruct full clustering results by loading individual components
            self.clustering_results = {}
            for emb_type in clustering_summary.keys():
                # Load labels
                labels_file = os.path.join(self.dirs['data'], f'cluster_labels_{emb_type}.npy')
                if os.path.exists(labels_file):
                    labels = np.load(labels_file)
                    
                    # Load model
                    model_file = os.path.join(self.dirs['models'], f'kmeans_model_{emb_type}.pkl')
                    model = None
                    if os.path.exists(model_file):
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                    
                    # Reconstruct full results
                    self.clustering_results[emb_type] = {
                        'k': clustering_summary[emb_type]['k'],
                        'labels': labels,
                        'silhouette_score': clustering_summary[emb_type]['silhouette_score'],
                        'silhouette_scores': clustering_summary[emb_type]['silhouette_scores'],
                        'k_values': clustering_summary[emb_type]['k_values'],
                        'model': model
                    }
                    
            print(f"Loaded clustering results for {len(self.clustering_results)} methods")
        
        # Load log
        log_file = os.path.join(self.dirs['reports'], 'processing_log.json')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.log = json.load(f)
            print(f"Loaded processing log: {len(self.log['steps_completed'])} steps completed")
        
        # Load saved models
        model_files = {
            'lda_model.pkl': 'lda',
            'lda_vectorizer.pkl': 'lda_vectorizer', 
            'tfidf_model.pkl': 'tfidf',
            'pca_model.pkl': 'pca'
        }
        
        for filename, model_name in model_files.items():
            model_path = os.path.join(self.dirs['models'], filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model")
        
        return len(self.embeddings), len(self.clustering_results)


# Enhanced usage example with full pipeline and output saving
def run_full_clustering_pipeline(data, text_column, label_column=None, sample_size=5000, output_dir='clustering_outputs'):
    """
    Run the complete clustering pipeline with output saving
    """
    print("="*60)
    print("STARTING COMPREHENSIVE TEXT CLUSTERING ANALYSIS")
    print("="*60)
    
    # Initialize clusterer
    clusterer = ContextAwareTextClustererWithOutputs(
        data=data, 
        text_column=text_column, 
        label_column=label_column,
        sample_size=sample_size,
        output_dir=output_dir
    )
    
    # Step 1: Create embeddings
    print("\nStep 1: Creating sentence embeddings...")
    clusterer.create_sentence_embeddings(['all-MiniLM-L6-v2'])
    
    print("\nStep 2: Creating Doc2Vec-like embeddings...")
    clusterer.create_doc2vec_like_embeddings()
    
    print("\nStep 3: Creating topic embeddings...")
    clusterer.create_topic_embeddings(n_topics=15)
    
    print("\nStep 4: Creating hybrid embeddings...")
    clusterer.create_hybrid_embeddings()
    
    # Step 2: Analyze semantic similarity
    print("\nStep 5: Analyzing semantic similarity...")
    for emb_type in clusterer.embeddings.keys():
        clusterer.semantic_similarity_analysis(embedding_type=emb_type, n_samples=100)
    
    # Step 3: Perform clustering
    print("\nStep 6: Performing contextual clustering...")
    results = clusterer.perform_contextual_clustering()
    
    # Step 4: Analyze clusters
    print("\nStep 7: Analyzing semantic clusters...")
    for emb_type in results.keys():
        clusterer.analyze_semantic_clusters(embedding_type=emb_type)
    
    # Step 5: Create visualizations
    print("\nStep 8: Creating visualizations...")
    for emb_type in results.keys():
        clusterer.visualize_semantic_clusters(embedding_type=emb_type, method='pca')
    
    # Step 6: Generate comprehensive report
    print("\nStep 9: Generating comprehensive report...")
    final_report = clusterer.generate_comprehensive_report()
    
    return clusterer, final_report


# Example usage with your data:
if __name__ == "__main__":
    # Load your data
    big_data = pd.read_csv("./nz_files/nz_filtered.csv")
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)       
        text = re.sub(r'\W+', ' ', text)      
        text = re.sub(r'\s+', ' ', text)      
        return text.strip()

    big_data["clean_text"] = big_data["text"].apply(clean_text)
    
    # Run the complete analysis pipeline
    clusterer, report = run_full_clustering_pipeline(
        data=big_data,
        text_column='text',
        label_column='party',
        sample_size=5000,
        output_dir='nz_clustering_analysis'
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Check the 'nz_clustering_analysis' folder for all outputs:")
    print("- plots/: All visualization plots")
    print("- data/: Processed data, embeddings, cluster labels")
    print("- models/: Trained models (LDA, PCA, KMeans, etc.)")
    print("- reports/: Analysis reports and summaries")
    print("- embeddings/: Raw embedding vectors")
    
    # Optional: Load results later
    # clusterer_loaded = ContextAwareTextClustererWithOutputs(data=big_data, text_column='text')
    # clusterer_loaded.load_saved_results('nz_clustering_analysis')