"""
Extractive summarization models using BERT embeddings and attention mechanisms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import networkx as nx
from .attention import DocumentAttention, MultiHeadAttention


class BERTExtractiveModel(nn.Module):
    """BERT-based extractive summarization with attention scoring."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        
        # Load pre-trained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze BERT parameters initially
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Document attention for sentence scoring
        self.document_attention = DocumentAttention(self.bert.config.hidden_size)
        
        # Sentence importance classifier
        self.importance_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def encode_sentences(self, sentences: List[str]) -> torch.Tensor:
        """Encode sentences using BERT."""
        sentence_embeddings = []
        
        for sentence in sentences:
            # Tokenize and encode
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.bert(**inputs)
                # Use [CLS] token embedding
                sentence_embedding = outputs.last_hidden_state[:, 0, :]
                sentence_embeddings.append(sentence_embedding)
        
        return torch.cat(sentence_embeddings, dim=0)
    
    def forward(self, sentences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for extractive summarization.
        
        Args:
            sentences: List of sentences in the document
            
        Returns:
            importance_scores: Importance score for each sentence
            attention_weights: Attention weights from document attention
        """
        # Encode sentences
        sentence_embeddings = self.encode_sentences(sentences)
        sentence_embeddings = sentence_embeddings.unsqueeze(0)  # Add batch dimension
        
        # Apply document attention
        document_repr, attention_weights = self.document_attention(sentence_embeddings)
        
        # Compute importance scores
        importance_scores = self.importance_classifier(sentence_embeddings.squeeze(0))
        importance_scores = importance_scores.squeeze(-1)
        
        return importance_scores, attention_weights.squeeze(0)
    
    def extract_summary(self, sentences: List[str], num_sentences: int = 3, 
                       threshold: float = 0.5) -> List[str]:
        """Extract summary sentences based on importance scores."""
        importance_scores, attention_weights = self.forward(sentences)
        
        # Get top sentences
        if num_sentences:
            top_indices = torch.topk(importance_scores, min(num_sentences, len(sentences))).indices
        else:
            top_indices = torch.where(importance_scores > threshold)[0]
        
        # Sort indices to maintain original order
        top_indices = sorted(top_indices.tolist())
        
        return [sentences[i] for i in top_indices]


class TextRankSummarizer:
    """TextRank algorithm for extractive summarization."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.sentence_model = SentenceTransformer(model_name)
    
    def build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Build sentence similarity matrix using sentence embeddings."""
        embeddings = self.sentence_model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Remove self-similarity
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
    
    def textrank_scores(self, similarity_matrix: np.ndarray, 
                       damping: float = 0.85, max_iter: int = 100, 
                       tol: float = 1e-4) -> np.ndarray:
        """Compute TextRank scores for sentences."""
        n = similarity_matrix.shape[0]
        
        # Create adjacency matrix
        adj_matrix = similarity_matrix / (similarity_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        # Initialize scores
        scores = np.ones(n) / n
        
        for _ in range(max_iter):
            new_scores = (1 - damping) / n + damping * adj_matrix.T.dot(scores)
            
            if np.abs(new_scores - scores).sum() < tol:
                break
                
            scores = new_scores
        
        return scores
    
    def extract_summary(self, sentences: List[str], num_sentences: int = 3) -> List[str]:
        """Extract summary using TextRank algorithm."""
        if len(sentences) <= num_sentences:
            return sentences
        
        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(sentences)
        
        # Compute TextRank scores
        scores = self.textrank_scores(similarity_matrix)
        
        # Get top sentences
        top_indices = np.argsort(scores)[-num_sentences:][::-1]
        top_indices = sorted(top_indices)  # Maintain original order
        
        return [sentences[i] for i in top_indices]


class ClusterBasedSummarizer:
    """Cluster-based extractive summarization."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.sentence_model = SentenceTransformer(model_name)
    
    def extract_summary(self, sentences: List[str], num_sentences: int = 3) -> List[str]:
        """Extract summary using clustering approach."""
        if len(sentences) <= num_sentences:
            return sentences
        
        # Encode sentences
        embeddings = self.sentence_model.encode(sentences)
        
        # Cluster sentences
        n_clusters = min(num_sentences, len(sentences))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Select representative sentence from each cluster
        summary_sentences = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            # Find sentence closest to cluster center
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            
            summary_sentences.append((closest_idx, sentences[closest_idx]))
        
        # Sort by original order
        summary_sentences.sort(key=lambda x: x[0])
        
        return [sent for _, sent in summary_sentences]


class HybridExtractiveModel:
    """Hybrid model combining multiple extractive approaches."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.bert_model = BERTExtractiveModel(model_name)
        self.textrank_model = TextRankSummarizer()
        self.cluster_model = ClusterBasedSummarizer()
    
    def extract_summary(self, sentences: List[str], num_sentences: int = 3, 
                       weights: Dict[str, float] = None) -> List[str]:
        """Extract summary using ensemble of methods."""
        if weights is None:
            weights = {"bert": 0.5, "textrank": 0.3, "cluster": 0.2}
        
        # Get scores from each method
        bert_scores, _ = self.bert_model.forward(sentences)
        bert_scores = bert_scores.detach().numpy()
        
        # TextRank scores
        similarity_matrix = self.textrank_model.build_similarity_matrix(sentences)
        textrank_scores = self.textrank_model.textrank_scores(similarity_matrix)
        
        # Cluster-based scores (distance to cluster centers)
        embeddings = self.cluster_model.sentence_model.encode(sentences)
        kmeans = KMeans(n_clusters=min(num_sentences, len(sentences)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        cluster_scores = np.zeros(len(sentences))
        
        for i, embedding in enumerate(embeddings):
            cluster_center = kmeans.cluster_centers_[clusters[i]]
            distance = np.linalg.norm(embedding - cluster_center)
            cluster_scores[i] = 1 / (1 + distance)  # Convert distance to similarity
        
        # Normalize scores
        bert_scores = (bert_scores - bert_scores.min()) / (bert_scores.max() - bert_scores.min() + 1e-8)
        textrank_scores = (textrank_scores - textrank_scores.min()) / (textrank_scores.max() - textrank_scores.min() + 1e-8)
        cluster_scores = (cluster_scores - cluster_scores.min()) / (cluster_scores.max() - cluster_scores.min() + 1e-8)
        
        # Combine scores
        final_scores = (weights["bert"] * bert_scores + 
                       weights["textrank"] * textrank_scores + 
                       weights["cluster"] * cluster_scores)
        
        # Get top sentences
        top_indices = np.argsort(final_scores)[-num_sentences:][::-1]
        top_indices = sorted(top_indices)  # Maintain original order
        
        return [sentences[i] for i in top_indices]