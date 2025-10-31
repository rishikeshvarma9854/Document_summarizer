"""
Custom attention mechanisms for document summarization.
Implements multi-head attention, self-attention, and document-level attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for document understanding."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(attention_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


class DocumentAttention(nn.Module):
    """Document-level attention for sentence importance scoring."""
    
    def __init__(self, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Sentence encoder
        self.sentence_encoder = nn.LSTM(
            hidden_size, hidden_size // 2, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=0.1
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, sentence_embeddings: torch.Tensor, 
                sentence_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sentence_embeddings: [batch_size, num_sentences, hidden_size]
            sentence_mask: [batch_size, num_sentences]
        
        Returns:
            document_representation: [batch_size, hidden_size]
            attention_weights: [batch_size, num_sentences]
        """
        # Encode sentences with LSTM
        lstm_output, _ = self.sentence_encoder(sentence_embeddings)
        lstm_output = self.dropout(lstm_output)
        
        # Compute attention scores
        attention_scores = self.attention(lstm_output).squeeze(-1)  # [batch_size, num_sentences]
        
        if sentence_mask is not None:
            attention_scores = attention_scores.masked_fill(sentence_mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute document representation
        document_representation = torch.sum(
            attention_weights.unsqueeze(-1) * lstm_output, dim=1
        )
        
        return document_representation, attention_weights


class HierarchicalAttention(nn.Module):
    """Hierarchical attention for word and sentence level understanding."""
    
    def __init__(self, word_hidden_size: int, sentence_hidden_size: int):
        super().__init__()
        
        # Word-level attention
        self.word_attention = MultiHeadAttention(word_hidden_size, num_heads=8)
        
        # Sentence-level attention
        self.sentence_attention = DocumentAttention(sentence_hidden_size)
        
        # Projection layers
        self.word_to_sentence = nn.Linear(word_hidden_size, sentence_hidden_size)
        
    def forward(self, word_embeddings: torch.Tensor, 
                word_mask: Optional[torch.Tensor] = None,
                sentence_boundaries: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            word_embeddings: [batch_size, num_words, word_hidden_size]
            word_mask: [batch_size, num_words]
            sentence_boundaries: [batch_size, num_sentences, 2] (start, end indices)
        
        Returns:
            document_representation: [batch_size, sentence_hidden_size]
            sentence_attention_weights: [batch_size, num_sentences]
            word_attention_weights: [batch_size, num_heads, num_words, num_words]
        """
        # Word-level attention
        word_attended, word_attention_weights = self.word_attention(
            word_embeddings, word_embeddings, word_embeddings, word_mask
        )
        
        # Aggregate words to sentences
        if sentence_boundaries is not None:
            sentence_embeddings = []
            for i in range(sentence_boundaries.size(1)):
                start_idx = sentence_boundaries[:, i, 0]
                end_idx = sentence_boundaries[:, i, 1]
                
                # Average pooling within sentence boundaries
                sentence_repr = []
                for b in range(word_attended.size(0)):
                    if end_idx[b] > start_idx[b]:
                        sent_words = word_attended[b, start_idx[b]:end_idx[b]]
                        sentence_repr.append(sent_words.mean(dim=0))
                    else:
                        sentence_repr.append(torch.zeros_like(word_attended[b, 0]))
                
                sentence_embeddings.append(torch.stack(sentence_repr))
            
            sentence_embeddings = torch.stack(sentence_embeddings, dim=1)
        else:
            # Simple chunking if no boundaries provided
            chunk_size = word_attended.size(1) // 10  # Assume 10 sentences
            sentence_embeddings = word_attended.view(
                word_attended.size(0), -1, chunk_size, word_attended.size(-1)
            ).mean(dim=2)
        
        # Project to sentence hidden size
        sentence_embeddings = self.word_to_sentence(sentence_embeddings)
        
        # Sentence-level attention
        document_representation, sentence_attention_weights = self.sentence_attention(
            sentence_embeddings
        )
        
        return document_representation, sentence_attention_weights, word_attention_weights