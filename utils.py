import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional



def load_embeddings(emb_type: str = "w2v") -> Dict[str, any]:
    """Load aligned embeddings from MUSE output"""
    # Add MUSE to path
    sys.path.insert(0, './MUSE')
    
    # Load embeddings
    en = torch.load(f"./embeddings/{emb_type}/aligned/vectors-en.pth", weights_only=False)
    fr = torch.load(f"./embeddings/{emb_type}/aligned/vectors-fr.pth", weights_only=False)
    return {
        "en_words": en["dico"],
        "en_vecs": en["vectors"].cpu().numpy(),
        "fr_words": fr["dico"],
        "fr_vecs": fr["vectors"].cpu().numpy()
    }


def cosine_sim(word1: str, word2: str, emb: Dict[str, any]) -> Optional[float]:
    """Compute cosine similarity between English word1 and French word2"""
    try:
        idx1 = emb["en_words"].index(word1)
        idx2 = emb["fr_words"].index(word2)
        vec1 = emb["en_vecs"][idx1].reshape(1, -1)
        vec2 = emb["fr_vecs"][idx2].reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]
    except:
        return None


def plot_pca(emb: Dict[str, any], word_pairs: List[Tuple[str, str]], title: str = "PCA Visualization") -> None:
    """Plot PCA for word pairs embeddings"""
    en_words, fr_words = [], []
    en_vecs, fr_vecs = [], []
    
    for en_word, fr_word in word_pairs:
        if en_word in emb["en_words"] and fr_word in emb["fr_words"]:
            en_idx = emb["en_words"].index(en_word)
            fr_idx = emb["fr_words"].index(fr_word)
            
            en_words.append(en_word)
            fr_words.append(fr_word)
            en_vecs.append(emb["en_vecs"][en_idx])
            fr_vecs.append(emb["fr_vecs"][fr_idx])
    
    all_vecs = np.vstack([en_vecs, fr_vecs])
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(all_vecs)
    
    n_en = len(en_words)
    en_coords = coords_2d[:n_en]
    fr_coords = coords_2d[n_en:]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(en_coords[:, 0], en_coords[:, 1], c='blue', label='English')
    plt.scatter(fr_coords[:, 0], fr_coords[:, 1], c='red', label='French')
    
    for i, word in enumerate(en_words):
        plt.annotate(word, (en_coords[i, 0], en_coords[i, 1]))
    
    for i, word in enumerate(fr_words):
        plt.annotate(word, (fr_coords[i, 0], fr_coords[i, 1]))
    
    plt.legend()
    plt.title(title)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_vocab(sentences: pd.Series, pattern: re.Pattern, max_vocab: Optional[int] = None) -> List[str]:
    """Get sorted unique vocabulary from text"""
    tokens = (
        sentences.fillna('')
        .str.lower()
        .apply(lambda x: pattern.findall(x))
        .explode()
        .dropna()
    )
    
    if max_vocab is None:
        return sorted(tokens.unique())
    else:
        word_counts = Counter(tokens)
        most_common = word_counts.most_common(max_vocab)
        top_words = [word for word, _ in most_common]
        return sorted(top_words)


def tokenize_corpus(series: pd.Series, pattern: re.Pattern) -> List[List[str]]:
    """Tokenize corpus into list of tokenized sentences"""
    sentences = (
        series.fillna('')
        .str.lower()
        .apply(lambda x: pattern.findall(x))
        .tolist()
    )
    return [s for s in sentences if len(s) > 0]


def save_embeddings_vec(vocab: List[str], embeddings: np.ndarray, output_file: str) -> None:
    """Save embeddings in word2vec format for MUSE"""
    kv = KeyedVectors(vector_size=embeddings.shape[1])
    kv.add_vectors(vocab, embeddings)
    kv.save_word2vec_format(output_file, binary=False)
