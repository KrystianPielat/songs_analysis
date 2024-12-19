from typing import List, Optional
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def correlation_heatmap(df: pd.DataFrame, features: List[str], title: str = 'Heatmap of Top Correlated Features', threshold: float = 0.6):
    num_df = df[features].dropna().select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()

    high_corr_features = corr_matrix[corr_matrix.abs() > threshold].stack().reset_index()
    high_corr_features = high_corr_features[high_corr_features['level_0'] != high_corr_features['level_1']]
    
    top_corr_features = pd.unique(high_corr_features[['level_0', 'level_1']].values.ravel('K'))

    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(num_df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title(title if title else 'Heatmap of Top Correlated Features', fontsize=12)
    plt.close(fig)  
    return fig

def plot_dendrogram(df: pd.DataFrame, features: List[str], title: str = "Features"):
    num_df = df[features].dropna().select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    distance_matrix = 1 - corr_matrix.abs()
    distance_matrix = distance_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)

    condensed_distance = distance_matrix.values[np.triu_indices_from(distance_matrix, k=1)]
    linked = linkage(condensed_distance, method='ward')

    fig, ax = plt.subplots(figsize=(15, 10))
    dendrogram(linked, labels=corr_matrix.columns, orientation='right', distance_sort='descending', ax=ax)
    ax.set_title(f'Hierarchical Clustering Dendrogram of {title}', fontsize=16)
    plt.close(fig)  
    return fig

def plot_network(df: pd.DataFrame, features: List[str], title: str = "Features", threshold: float = 0.4):
    num_df = df[features].dropna().select_dtypes(include=[np.number])
    corr_matrix = num_df.corr().abs()
    high_corr_pairs = corr_matrix.unstack().reset_index()
    high_corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']
    high_corr_pairs = high_corr_pairs[(high_corr_pairs['Correlation'] > threshold) & (high_corr_pairs['Feature1'] != high_corr_pairs['Feature2'])]

    G = nx.from_pandas_edgelist(high_corr_pairs, 'Feature1', 'Feature2', edge_attr='Correlation')
    fig, ax = plt.subplots(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.15)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='gray', font_size=10, ax=ax)
    ax.set_title(f'Network Graph of Highly Correlated {title}', fontsize=16)
    plt.close(fig)  
    return fig

def plot_tsne(df: pd.DataFrame, features: List[str], hue: Optional[str] = None, title: str = "Feature"):
    num_df = df[features].dropna().select_dtypes(include=[np.number])
    if hue:
        hue_values = df.loc[num_df.index, hue]
    else:
        hue_values = None

    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(num_df)

    tsne_df = pd.DataFrame(tsne_result, columns=['Component 1', 'Component 2'])
    if hue:
        tsne_df['Hue'] = hue_values.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        x='Component 1', 
        y='Component 2', 
        hue='Hue' if hue else None, 
        data=tsne_df, 
        alpha=0.7, 
        ax=ax
    )
    ax.set_title(f't-SNE Visualization of {title} Relationships', fontsize=16)
    plt.close(fig)  
    return fig


def analyze_genre_features(df, features):
    df_numeric = df[features].select_dtypes(include=[float, int])
    features = df_numeric.columns.tolist()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=features, index=df.index)
    df_scaled['genre'] = df['genre']
    genre_means = df_scaled.groupby('genre')[features].mean()
    plt.figure(figsize=(15, 10))
    heatmap_figure = sns.heatmap(genre_means, annot=False, cmap='coolwarm', cbar_kws={'label': 'Scaled Mean Value'}).get_figure()
    plt.title('Scaled Mean Feature Values by Genre', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Genres', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.close(heatmap_figure)
    top_5_features_by_genre = genre_means.apply(lambda x: x.nlargest(5).index.tolist(), axis=1)
    top_5_features_df = top_5_features_by_genre.apply(pd.Series)
    top_5_features_df.columns = [f"Top {i+1}" for i in range(top_5_features_df.shape[1])]
    top_5_features_df = top_5_features_df.reset_index().rename(columns={'index': 'Genre'})
    return top_5_features_df, heatmap_figure

def plot_feature_max_by_genre_heatmap(df, features):
    df_numeric = df[features].select_dtypes(include=[float, int])
    features = df_numeric.columns.tolist()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=features, index=df.index)
    df_scaled['genre'] = df['genre']
    genre_means = df_scaled.groupby('genre')[features].mean()
    highlight_mask = genre_means == genre_means.max(axis=0)
    plt.figure(figsize=(15, 10))
    heatmap_figure = sns.heatmap(
        genre_means,
        mask=~highlight_mask,
        annot=True,
        cmap='coolwarm',
        cbar=False,
        linewidths=0.5,
        linecolor='gray'
    ).get_figure()
    plt.title('Heatmap Highlighting Maximum Feature Values by Genre', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Genres', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.close(heatmap_figure)
    return heatmap_figure

def plot_top_features_per_genre_heatmap(df, features):
    df_numeric = df[features].select_dtypes(include=[float, int])
    features = df_numeric.columns.tolist()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=features, index=df.index)
    df_scaled['genre'] = df['genre']
    genre_means = df_scaled.groupby('genre')[features].mean()
    top_5_mask = genre_means.apply(lambda x: x >= x.nlargest(5).min(), axis=1)
    plt.figure(figsize=(15, 10))
    heatmap_figure = sns.heatmap(
        genre_means,
        mask=~top_5_mask,
        annot=True,
        cmap='coolwarm',
        cbar=False,
        linewidths=0.5,
        linecolor='gray'
    ).get_figure()
    plt.title('Heatmap Highlighting Top 5 Features by Genre', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Genres', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.close(heatmap_figure)
    return heatmap_figure
