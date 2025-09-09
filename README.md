# MP Speeches Analysis Project

A comprehensive computational analysis of parliamentary speeches across multiple countries, focusing on gender differences, sentiment analysis, and text clustering patterns in political discourse.

## Overview

This project analyzes parliamentary speech data from three countries to understand:
- **Gender differences** in political speech patterns and sentiment
- **Party-based** discourse variations
- **Temporal trends** in political sentiment (2000 onwards)
- **Semantic clustering** of political speeches using multiple embedding techniques

## Countries & Datasets Analyzed

### 1. New Zealand (NZ)
- **Dataset**: New Zealand House of Representatives speeches
- **Source**: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN)
- **Time Period**: 2000 onwards
- **Focus**: Gender sentiment analysis, party discourse patterns

### 2. Germany (Bundestag)
- **Dataset**: German Bundestag parliamentary speeches
- **Source**: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN)
- **Time Period**: 2000 onwards
- **Focus**: Gender sentiment analysis, party discourse patterns

### 3. Netherlands (Tweede Kamer)
- **Dataset**: Dutch Tweede Kamer parliamentary speeches
- **Source**: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN)
- **Time Period**: 2000 onwards
- **Focus**: Sentiment analysis and discourse patterns

## Project Structure

```
MP_speeches/
├── bin/                                    # Binary files
├── bundestag_files/                        # German parliamentary data
├── dutch_files/                           # Dutch parliamentary data
├── nz_files/                              # New Zealand parliamentary data
├── embedding_outputs/                      # Python embedding analysis outputs
│   ├── data/                              # Processed datasets
│   ├── embeddings/                         # Generated embeddings
│   └── plots/                             # Visualization outputs
├── nz_clustering_analysis/                 # Comprehensive clustering analysis
│   ├── data/                              # Cluster labels and similarity matrices
│   ├── embeddings/                         # Multiple embedding types
│   ├── models/                            # Trained ML models
│   ├── plots/                             # Clustering visualizations
│   └── reports/                           # Analysis reports
├── gender_EDA_bundestag/                  # German gender analysis outputs
├── gender_EDA_nz/                         # NZ gender analysis outputs
└── [Analysis Scripts]                      # R and Python analysis files
```

## Key Analyses

### 1. Gender & Sentiment Analysis
- **Sentiment Scoring**: Using Rauh dictionary for political sentiment analysis
- **Gender Classification**: Automated gender prediction from speaker names
- **Statistical Modeling**: Linear regression analysis of gender-sentiment relationships
- **Visualization**: Temporal trends and party-based comparisons

### 2. Text Clustering & Embeddings
- **Multiple Embedding Types**:
  - Sentence Transformers (all-MiniLM-L6-v2)
  - Doc2Vec-like embeddings
  - Topic-based embeddings (LDA)
  - Hybrid embeddings combining multiple approaches
- **Clustering Methods**: K-means clustering with semantic analysis
- **Dimensionality Reduction**: PCA and UMAP for visualization
- **Similarity Analysis**: Semantic similarity matrices and heatmaps

### 3. Party Discourse Analysis
- **TF-IDF Analysis**: Term frequency analysis by party
- **Topic Modeling**: Latent Dirichlet Allocation for topic discovery
- **Cross-party Comparisons**: Statistical analysis of discourse differences

## Technical Stack

### Python Environment
- **Core Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **NLP Libraries**: nltk, spacy, sentence-transformers, gensim
- **Clustering**: umap-learn, hdbscan
- **Environment**: Conda environment specified in `environment.yaml`

### R Environment
- **Core Libraries**: tidyverse, dplyr, ggplot2
- **NLP Libraries**: quanteda, spacyr, quanteda.sentiment
- **Statistical Analysis**: car, leaps, robustbase, emmeans
- **Gender Analysis**: gender, genderdata

## Getting Started

### Prerequisites
1. **Python Environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate text-cluster
   ```

2. **R Environment**:
   - Install required R packages as specified in analysis scripts
   - Ensure quanteda and related packages are installed

3. **Data Access**:
   - Download datasets from Harvard Dataverse
   - Place data files in appropriate directories (`bundestag_files/`, `nz_files/`, `dutch_files/`)

### Running Analyses

#### Gender & Sentiment Analysis
```r
# New Zealand analysis
source("N00_NZ_gender_sentiment_EDA.R")
source("N01_NZ_linear_reg.R")

# German Bundestag analysis  
source("G00_bundestag_gender_sentiment_EDA.R")
source("G01_bundestag_linear_reg.R")

# Dutch Tweede Kamer analysis
source("D00_tweede.R")
```

#### Text Clustering Analysis
```python
# Generate embeddings and perform clustering
python generate_embeddings.py

# Run comprehensive clustering pipeline
python test.py

# Run embeddings clustering analysis
python embeddings_clustering.py
```

## Key Findings & Outputs

### 1. Gender Analysis Results
- **Sentiment Differences**: Analysis of gender-based sentiment patterns across parties
- **Statistical Models**: Robust regression models examining gender-sentiment relationships
- **Temporal Trends**: Evolution of gender differences in political discourse over time

### 2. Clustering Analysis Results
- **Semantic Clusters**: Identification of thematic speech clusters
- **Party Alignment**: Analysis of how political parties align with semantic clusters
- **Similarity Networks**: Visualization of speech similarity patterns

### 3. Visualization Outputs
- **Interactive Plots**: UMAP visualizations with party-based coloring
- **Heatmaps**: Similarity matrices showing speech relationships
- **Statistical Plots**: Box plots, scatter plots, and trend analyses

## Data Processing Pipeline

1. **Data Cleaning**: Filter speeches with >50 terms, remove independent MPs
2. **Gender Classification**: Automated gender prediction from speaker names
3. **Sentiment Analysis**: Rauh dictionary-based sentiment scoring
4. **Text Preprocessing**: Tokenization, lemmatization, stopword removal
5. **Embedding Generation**: Multiple embedding approaches for comprehensive analysis
6. **Clustering**: K-means clustering with semantic analysis
7. **Visualization**: Dimensionality reduction and interactive plotting

## Research Applications

This project provides insights into:
- **Political Communication**: How gender influences political discourse
- **Party Dynamics**: Cross-party discourse differences and similarities
- **Temporal Analysis**: Evolution of political speech patterns over time
- **Computational Methods**: Advanced NLP techniques for political text analysis

## Contributing

This is a research project analyzing parliamentary speech data. Contributions should focus on:
- Improving analysis methodologies
- Adding new countries/datasets
- Enhancing visualization techniques
- Statistical model improvements

## License

This project uses publicly available parliamentary speech data from Harvard Dataverse. Please ensure compliance with data usage terms when using this code.

## Contact

For questions about this analysis or collaboration opportunities, please refer to the project documentation and analysis scripts.

