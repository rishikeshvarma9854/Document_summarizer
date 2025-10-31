"""
Text preprocessing utilities for document summarization.
Handles text cleaning, sentence segmentation, and tokenization.
"""

import re
import string
import nltk
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Try to import spaCy, but don't fail if it's not available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


class TextPreprocessor:
    """Main text preprocessing class for summarization tasks."""
    
    def __init__(self, language: str = "english", use_spacy: bool = True):
        self.language = language
        self.use_spacy = use_spacy
        
        # Initialize NLTK components
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
        # Initialize spaCy if requested and available
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
                self.nlp = None
        elif use_spacy and not SPACY_AVAILABLE:
            print("spaCy not available. Using NLTK instead.")
            self.use_spacy = False
            self.nlp = None
        else:
            self.nlp = None
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                nltk.download(data, quiet=True)
    
    def clean_text(self, text: str, remove_special_chars: bool = True,
                   remove_digits: bool = False, lowercase: bool = True) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters if requested
        if remove_special_chars:
            text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        # Remove digits if requested
        if remove_digits:
            text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        return text.strip()
    
    def segment_sentences(self, text: str, min_length: int = 10) -> List[str]:
        """Segment text into sentences with filtering."""
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            sentences = sent_tokenize(text)
        
        # Filter sentences by minimum length
        sentences = [sent for sent in sentences if len(sent.split()) >= min_length]
        
        return sentences
    
    def tokenize_words(self, text: str, remove_stopwords: bool = True,
                      apply_stemming: bool = False, apply_lemmatization: bool = True) -> List[str]:
        """Tokenize text into words with optional preprocessing."""
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        else:
            tokens = word_tokenize(text.lower())
            tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if apply_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization
        if apply_lemmatization and self.use_spacy and self.nlp:
            doc = self.nlp(' '.join(tokens))
            tokens = [token.lemma_ for token in doc]
        elif apply_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF scoring."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        sentences = self.segment_sentences(text)
        if len(sentences) < 2:
            return []
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]
            
            return keywords
        except ValueError:
            return []
    
    def get_sentence_features(self, sentences: List[str]) -> Dict[str, List[float]]:
        """Extract features for each sentence."""
        features = {
            'length': [],
            'position': [],
            'keyword_density': [],
            'numeric_count': [],
            'capital_ratio': []
        }
        
        # Extract keywords from all sentences combined
        all_text = ' '.join(sentences)
        keywords = [kw[0] for kw in self.extract_keywords(all_text, top_k=20)]
        keyword_set = set(keywords)
        
        for i, sentence in enumerate(sentences):
            # Length feature
            features['length'].append(len(sentence.split()))
            
            # Position feature (normalized)
            features['position'].append(i / len(sentences))
            
            # Keyword density
            words = self.tokenize_words(sentence, remove_stopwords=True)
            keyword_count = sum(1 for word in words if word in keyword_set)
            features['keyword_density'].append(keyword_count / max(len(words), 1))
            
            # Numeric count
            numeric_count = len(re.findall(r'\d+', sentence))
            features['numeric_count'].append(numeric_count)
            
            # Capital letter ratio
            capital_count = sum(1 for c in sentence if c.isupper())
            features['capital_ratio'].append(capital_count / max(len(sentence), 1))
        
        return features


class DocumentProcessor:
    """Process documents for summarization tasks."""
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def process_document(self, text: str, max_sentences: int = 100) -> Dict:
        """Process a single document for summarization."""
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Segment into sentences
        sentences = self.preprocessor.segment_sentences(cleaned_text)
        
        # Limit number of sentences
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        # Extract features
        features = self.preprocessor.get_sentence_features(sentences)
        
        # Extract keywords
        keywords = self.preprocessor.extract_keywords(cleaned_text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'features': features,
            'keywords': keywords,
            'num_sentences': len(sentences),
            'num_words': len(cleaned_text.split())
        }
    
    def process_batch(self, documents: List[str], max_sentences: int = 100) -> List[Dict]:
        """Process multiple documents."""
        return [self.process_document(doc, max_sentences) for doc in documents]
    
    def create_dataset(self, documents: List[str], summaries: Optional[List[str]] = None) -> pd.DataFrame:
        """Create a dataset for training/evaluation."""
        processed_docs = self.process_batch(documents)
        
        data = []
        for i, doc_data in enumerate(processed_docs):
            row = {
                'document_id': i,
                'original_text': doc_data['original_text'],
                'cleaned_text': doc_data['cleaned_text'],
                'num_sentences': doc_data['num_sentences'],
                'num_words': doc_data['num_words'],
                'keywords': doc_data['keywords'][:5]  # Top 5 keywords
            }
            
            if summaries and i < len(summaries):
                row['summary'] = summaries[i]
                row['summary_length'] = len(summaries[i].split())
            
            data.append(row)
        
        return pd.DataFrame(data)


class SummarizationDataLoader:
    """Data loader for summarization datasets."""
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.document_processor = DocumentProcessor(self.preprocessor)
    
    def load_cnn_dailymail(self, split: str = "train", max_samples: int = 1000) -> pd.DataFrame:
        """Load CNN/DailyMail dataset."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            documents = [item['article'] for item in dataset]
            summaries = [item['highlights'] for item in dataset]
            
            return self.document_processor.create_dataset(documents, summaries)
            
        except ImportError:
            print("datasets library not installed. Install with: pip install datasets")
            return pd.DataFrame()
    
    def load_xsum(self, split: str = "train", max_samples: int = 1000) -> pd.DataFrame:
        """Load XSum dataset."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("xsum", split=split)
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            documents = [item['document'] for item in dataset]
            summaries = [item['summary'] for item in dataset]
            
            return self.document_processor.create_dataset(documents, summaries)
            
        except ImportError:
            print("datasets library not installed. Install with: pip install datasets")
            return pd.DataFrame()
    
    def load_custom_data(self, file_path: str, text_column: str = "text", 
                        summary_column: Optional[str] = None) -> pd.DataFrame:
        """Load custom dataset from file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            documents = df[text_column].tolist()
            summaries = df[summary_column].tolist() if summary_column and summary_column in df.columns else None
            
            return self.document_processor.create_dataset(documents, summaries)
            
        except Exception as e:
            print(f"Error loading custom data: {e}")
            return pd.DataFrame()


def preprocess_for_model(text: str, model_type: str = "bert", max_length: int = 512) -> Dict:
    """Preprocess text for specific model types."""
    preprocessor = TextPreprocessor()
    
    if model_type.lower() in ["bert", "roberta", "distilbert"]:
        # For BERT-like models
        cleaned_text = preprocessor.clean_text(text, remove_special_chars=False)
        sentences = preprocessor.segment_sentences(cleaned_text)
        
        return {
            'text': cleaned_text[:max_length * 4],  # Rough character limit
            'sentences': sentences,
            'num_sentences': len(sentences)
        }
    
    elif model_type.lower() in ["t5", "bart", "pegasus"]:
        # For seq2seq models
        cleaned_text = preprocessor.clean_text(text)
        
        # Add model-specific prefixes
        if model_type.lower() == "t5":
            cleaned_text = f"summarize: {cleaned_text}"
        
        return {
            'text': cleaned_text[:max_length * 4],
            'sentences': preprocessor.segment_sentences(cleaned_text),
            'model_input': cleaned_text
        }
    
    else:
        # Default preprocessing
        return {
            'text': preprocessor.clean_text(text),
            'sentences': preprocessor.segment_sentences(text)
        }