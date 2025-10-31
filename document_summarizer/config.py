"""
Configuration file for the Document Summarizer application.
"""

import os
import streamlit as st

class Config:
    """Configuration class for managing app settings."""
    
    # Hugging Face API Configuration
    HUGGINGFACE_API_KEY = None  # Set via environment variable or user input
    
    # Model Configuration
    DEFAULT_MODEL = "t5"  # Changed from textrank to t5
    AVAILABLE_MODELS = {
        "t5": "t5-small",
        "bart": "facebook/bart-large-cnn", 
        "flan-t5": "google/flan-t5-small"
    }
    
    # Summary Configuration
    DEFAULT_COMPRESSION_RATIO = 0.3
    MIN_SUMMARY_LENGTH = 30
    MAX_SUMMARY_LENGTH = 200
    
    # Text Processing Configuration
    MAX_CHUNK_SIZE = 3000
    MAX_CHUNKS = 3
    
    @classmethod
    def get_hf_token(cls):
        """Get Hugging Face API token from various sources."""
        # Try Streamlit secrets first (for deployment)
        try:
            if hasattr(st, 'secrets') and 'HUGGINGFACE_API_KEY' in st.secrets:
                return st.secrets['HUGGINGFACE_API_KEY']
        except:
            pass
            
        # Try environment variable
        env_token = os.getenv('HUGGINGFACE_API_KEY')
        if env_token:
            return env_token
            
        # Try session state (user input)
        if hasattr(st, 'session_state') and 'hf_token' in st.session_state:
            return st.session_state['hf_token']
        
        # Return hardcoded token if set
        if cls.HUGGINGFACE_API_KEY:
            return cls.HUGGINGFACE_API_KEY
            
        return None
    
    @classmethod
    def set_hf_token(cls, token):
        """Set Hugging Face API token."""
        if hasattr(st, 'session_state'):
            st.session_state['hf_token'] = token
        cls.HUGGINGFACE_API_KEY = token
    
    @classmethod
    def get_model_config(cls, model_name):
        """Get model configuration."""
        return cls.AVAILABLE_MODELS.get(model_name, cls.AVAILABLE_MODELS[cls.DEFAULT_MODEL])
