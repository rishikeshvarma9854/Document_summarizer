"""
Abstractive summarization models using transformer architectures.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    PegasusForConditionalGeneration, PegasusTokenizer,
    GenerationConfig
)
from .attention import MultiHeadAttention


class T5Summarizer:
    """T5-based abstractive summarization model."""
    
    def __init__(self, model_name: str = "t5-small", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Generation configuration
        self.generation_config = GenerationConfig(
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    def generate_summary(self, text: str, max_summary_length: int = 150) -> str:
        """Generate abstractive summary using T5."""
        # Prepare input with T5 prefix
        input_text = f"summarize: {text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Update generation config
        self.generation_config.max_length = max_summary_length
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.generation_config
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()


class BARTSummarizer:
    """BART-based abstractive summarization model."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        self.generation_config = GenerationConfig(
            max_length=142,
            min_length=56,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    def generate_summary(self, text: str, max_summary_length: int = 142) -> str:
        """Generate abstractive summary using BART."""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        # Update generation config
        self.generation_config.max_length = max_summary_length
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.generation_config
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()


class PegasusSummarizer:
    """Pegasus-based abstractive summarization model."""
    
    def __init__(self, model_name: str = "google/pegasus-xsum"):
        self.model_name = model_name
        
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        
        self.generation_config = GenerationConfig(
            max_length=64,
            min_length=8,
            length_penalty=0.8,
            num_beams=8,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    def generate_summary(self, text: str, max_summary_length: int = 64) -> str:
        """Generate abstractive summary using Pegasus."""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Update generation config
        self.generation_config.max_length = max_summary_length
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.generation_config
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()


class CustomTransformerSummarizer(nn.Module):
    """Custom transformer model for abstractive summarization."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the transformer."""
        
        # Embeddings
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Output projection
        return self.output_projection(output)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate square subsequent mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class HybridAbstractiveModel:
    """Hybrid model combining multiple abstractive approaches."""
    
    def __init__(self):
        self.t5_model = T5Summarizer("t5-small")
        self.bart_model = BARTSummarizer("facebook/bart-large-cnn")
        self.pegasus_model = PegasusSummarizer("google/pegasus-xsum")
    
    def generate_summary(self, text: str, method: str = "ensemble", 
                        max_summary_length: int = 150) -> str:
        """Generate summary using specified method or ensemble."""
        
        if method == "t5":
            return self.t5_model.generate_summary(text, max_summary_length)
        elif method == "bart":
            return self.bart_model.generate_summary(text, max_summary_length)
        elif method == "pegasus":
            return self.pegasus_model.generate_summary(text, max_summary_length)
        elif method == "ensemble":
            # Generate summaries from all models
            t5_summary = self.t5_model.generate_summary(text, max_summary_length)
            bart_summary = self.bart_model.generate_summary(text, max_summary_length)
            pegasus_summary = self.pegasus_model.generate_summary(text, max_summary_length)
            
            # Simple ensemble: return the longest summary (often more informative)
            summaries = [t5_summary, bart_summary, pegasus_summary]
            return max(summaries, key=len)
        else:
            raise ValueError(f"Unknown method: {method}")


import math