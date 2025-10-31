"""
Main summarization pipeline integrating extractive and abstractive models.
"""

import torch
from typing import List, Dict, Optional, Union
try:
    from models.extractive import BERTExtractiveModel, TextRankSummarizer, HybridExtractiveModel
    from models.abstractive import T5Summarizer, BARTSummarizer, HybridAbstractiveModel
    from utils.preprocessing import TextPreprocessor, DocumentProcessor
    from utils.evaluation import SummarizationEvaluator
except ImportError:
    # Fallback for when running from parent directory
    try:
        from .models.extractive import BERTExtractiveModel, TextRankSummarizer, HybridExtractiveModel
        from .models.abstractive import T5Summarizer, BARTSummarizer, HybridAbstractiveModel
        from .utils.preprocessing import TextPreprocessor, DocumentProcessor
        from .utils.evaluation import SummarizationEvaluator
    except ImportError:
        # If all else fails, set to None and handle gracefully
        BERTExtractiveModel = None
        TextRankSummarizer = None
        HybridExtractiveModel = None
        T5Summarizer = None
        BARTSummarizer = None
        HybridAbstractiveModel = None
        TextPreprocessor = None
        DocumentProcessor = None
        SummarizationEvaluator = None


class SummarizationPipeline:
    """Complete summarization pipeline with multiple approaches."""
    
    def __init__(self, approach: str = "hybrid", device: str = "auto"):
        self.approach = approach.lower()
        self.device = self._get_device(device)
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        self.document_processor = DocumentProcessor(self.preprocessor)
        
        # Initialize models based on approach
        self.extractive_model = None
        self.abstractive_model = None
        
        if self.approach in ["extractive", "hybrid"]:
            self._load_extractive_models()
        
        if self.approach in ["abstractive", "hybrid"]:
            self._load_abstractive_models()
        
        # Initialize evaluator
        self.evaluator = SummarizationEvaluator()
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_extractive_models(self):
        """Load extractive summarization models."""
        if HybridExtractiveModel is None:
            print("⚠️ Extractive models not available. Using QuickSummarizer fallback.")
            self.extractive_model = None
            return
        
        try:
            print("Loading extractive models...")
            self.extractive_model = HybridExtractiveModel()
            if hasattr(self.extractive_model.bert_model, 'to'):
                self.extractive_model.bert_model.to(self.device)
        except Exception as e:
            print(f"⚠️ Failed to load extractive models: {e}")
            self.extractive_model = None
    
    def _load_abstractive_models(self):
        """Load abstractive summarization models."""
        if HybridAbstractiveModel is None:
            print("⚠️ Abstractive models not available. Using QuickSummarizer fallback.")
            self.abstractive_model = None
            return
        
        try:
            print("Loading abstractive models...")
            self.abstractive_model = HybridAbstractiveModel()
        except Exception as e:
            print(f"⚠️ Failed to load abstractive models: {e}")
            self.abstractive_model = None
    
    def summarize(self, text: str, method: str = "auto", max_length: int = 150, 
                 num_sentences: int = 3) -> Dict[str, str]:
        """
        Generate summary using specified method.
        
        Args:
            text: Input text to summarize
            method: Summarization method ('extractive', 'abstractive', 'hybrid', 'auto')
            max_length: Maximum length for abstractive summaries
            num_sentences: Number of sentences for extractive summaries
            
        Returns:
            Dictionary with summary and metadata
        """
        if method == "auto":
            method = self.approach
        
        # Preprocess text
        processed_doc = self.document_processor.process_document(text)
        sentences = processed_doc['sentences']
        
        results = {
            'original_length': len(text.split()),
            'num_sentences': len(sentences),
            'method': method
        }
        
        if method == "extractive" and self.extractive_model:
            summary = self._extractive_summarize(sentences, num_sentences)
            results['summary'] = summary
            results['summary_type'] = 'extractive'
            
        elif method == "abstractive" and self.abstractive_model:
            summary = self._abstractive_summarize(text, max_length)
            results['summary'] = summary
            results['summary_type'] = 'abstractive'
            
        elif method == "hybrid":
            # Generate both types and combine
            extractive_summary = self._extractive_summarize(sentences, num_sentences) if self.extractive_model else ""
            abstractive_summary = self._abstractive_summarize(text, max_length) if self.abstractive_model else ""
            
            # Simple hybrid approach: use abstractive if available, otherwise extractive
            if abstractive_summary and len(abstractive_summary.strip()) > 0:
                summary = abstractive_summary
                results['summary_type'] = 'abstractive'
            else:
                summary = extractive_summary
                results['summary_type'] = 'extractive'
            
            results['summary'] = summary
            results['extractive_summary'] = extractive_summary
            results['abstractive_summary'] = abstractive_summary
            
        else:
            raise ValueError(f"Unknown method: {method} or models not loaded")
        
        # Calculate compression ratio
        results['compression_ratio'] = len(results['summary'].split()) / results['original_length']
        
        return results
    
    def _extractive_summarize(self, sentences: List[str], num_sentences: int) -> str:
        """Generate extractive summary."""
        if not sentences:
            return ""
        
        try:
            summary_sentences = self.extractive_model.extract_summary(sentences, num_sentences)
            return ' '.join(summary_sentences)
        except Exception as e:
            print(f"Extractive summarization failed: {e}")
            # Fallback to simple approach
            return ' '.join(sentences[:min(num_sentences, len(sentences))])
    
    def _abstractive_summarize(self, text: str, max_length: int) -> str:
        """Generate abstractive summary."""
        if not text.strip():
            return ""
        
        try:
            return self.abstractive_model.generate_summary(text, method="ensemble", max_summary_length=max_length)
        except Exception as e:
            print(f"Abstractive summarization failed: {e}")
            # Fallback to extractive
            sentences = self.preprocessor.segment_sentences(text)
            return ' '.join(sentences[:3]) if sentences else ""
    
    def batch_summarize(self, texts: List[str], method: str = "auto", 
                       max_length: int = 150, num_sentences: int = 3) -> List[Dict[str, str]]:
        """Summarize multiple texts."""
        return [self.summarize(text, method, max_length, num_sentences) for text in texts]
    
    def evaluate_summary(self, original_text: str, generated_summary: str, 
                        reference_summary: Optional[str] = None) -> Dict[str, float]:
        """Evaluate generated summary."""
        if reference_summary:
            return self.evaluator.evaluate_single(reference_summary, generated_summary, original_text)
        else:
            # Self-evaluation metrics
            return {
                'compression_ratio': self.evaluator.custom_evaluator.compression_ratio(original_text, generated_summary),
                'coverage_score': self.evaluator.custom_evaluator.coverage_score(original_text, generated_summary),
                'readability_score': self.evaluator.custom_evaluator.readability_score(generated_summary)
            }
    
    def compare_methods(self, text: str, max_length: int = 150, num_sentences: int = 3) -> Dict[str, Dict]:
        """Compare different summarization methods on the same text."""
        results = {}
        
        if self.extractive_model:
            extractive_result = self.summarize(text, "extractive", max_length, num_sentences)
            results['extractive'] = extractive_result
        
        if self.abstractive_model:
            abstractive_result = self.summarize(text, "abstractive", max_length, num_sentences)
            results['abstractive'] = abstractive_result
        
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about loaded models."""
        info = {
            'approach': self.approach,
            'device': str(self.device),
            'extractive_models': [],
            'abstractive_models': []
        }
        
        if self.extractive_model:
            info['extractive_models'] = ['BERT-based', 'TextRank', 'Clustering']
        
        if self.abstractive_model:
            info['abstractive_models'] = ['T5', 'BART', 'Pegasus']
        
        return info


class QuickSummarizer:
    """Lightweight summarizer for quick results using simple TextRank."""
    
    def __init__(self):
        if TextPreprocessor is not None:
            self.preprocessor = TextPreprocessor()
        else:
            # Create a minimal preprocessor if import failed
            self.preprocessor = self._create_minimal_preprocessor()
    
    def _create_minimal_preprocessor(self):
        """Create a minimal preprocessor when imports fail."""
        class MinimalPreprocessor:
            def clean_text(self, text):
                import re
                # Basic cleaning
                text = re.sub(r'\s+', ' ', text).strip()
                text = re.sub(r'http[s]?://\S+', '', text)
                text = re.sub(r'\S+@\S+', '', text)
                return text.lower()
            
            def segment_sentences(self, text):
                import re
                # Simple sentence splitting
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                return sentences
        
        return MinimalPreprocessor()
    
    def _tokenize(self, text):
        """Simple tokenization."""
        import re
        return re.findall(r'\w+', text.lower())
    
    def _compute_tf_idf(self, sentences):
        """Compute TF-IDF vectors for sentences."""
        from collections import Counter
        import numpy as np
        
        tokenized_sentences = [self._tokenize(sent) for sent in sentences]
        
        # Build vocabulary
        vocab = set()
        for tokens in tokenized_sentences:
            vocab.update(tokens)
        vocab = list(vocab)
        vocab_to_idx = {word: i for i, word in enumerate(vocab)}
        
        # Compute TF-IDF
        tfidf_vectors = []
        
        # Document frequency
        df = Counter()
        for tokens in tokenized_sentences:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
        
        for tokens in tokenized_sentences:
            # Term frequency
            tf = Counter(tokens)
            total_terms = len(tokens)
            
            # TF-IDF vector
            vector = np.zeros(len(vocab))
            for token, count in tf.items():
                if token in vocab_to_idx:
                    tf_score = count / total_terms
                    idf_score = np.log(len(sentences) / (df[token] + 1))
                    vector[vocab_to_idx[token]] = tf_score * idf_score
            
            tfidf_vectors.append(vector)
        
        return np.array(tfidf_vectors)
    
    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def _build_similarity_matrix(self, sentences):
        """Build similarity matrix between sentences."""
        import numpy as np
        tfidf_vectors = self._compute_tf_idf(sentences)
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity_matrix[i][j] = self._cosine_similarity(
                        tfidf_vectors[i], tfidf_vectors[j]
                    )
        
        return similarity_matrix
    
    def _textrank_scores(self, similarity_matrix, damping=0.85, max_iter=100, tol=1e-4):
        """Compute TextRank scores."""
        import numpy as np
        n = similarity_matrix.shape[0]
        
        # Normalize similarity matrix
        row_sums = similarity_matrix.sum(axis=1)
        normalized_matrix = np.zeros_like(similarity_matrix)
        
        for i in range(n):
            if row_sums[i] > 0:
                normalized_matrix[i] = similarity_matrix[i] / row_sums[i]
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Iterate
        for _ in range(max_iter):
            new_scores = (1 - damping) / n + damping * normalized_matrix.T.dot(scores)
            
            if np.abs(new_scores - scores).sum() < tol:
                break
            
            scores = new_scores
        
        return scores
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Quick extractive summarization using TextRank."""
        import numpy as np
        
        # Clean and segment text
        cleaned_text = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.segment_sentences(cleaned_text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        try:
            # Build similarity matrix and compute scores
            similarity_matrix = self._build_similarity_matrix(sentences)
            scores = self._textrank_scores(similarity_matrix)
            
            # Get top sentences
            top_indices = np.argsort(scores)[-num_sentences:][::-1]
            top_indices = sorted(top_indices)  # Maintain original order
            
            summary_sentences = [sentences[i] for i in top_indices]
            return ' '.join(summary_sentences)
        except Exception as e:
            print(f"TextRank failed, using first {num_sentences} sentences: {e}")
            return ' '.join(sentences[:num_sentences])


class BatchProcessor:
    """Process multiple documents efficiently."""
    
    def __init__(self, pipeline: SummarizationPipeline):
        self.pipeline = pipeline
    
    def process_documents(self, documents: List[str], batch_size: int = 8, 
                         method: str = "auto") -> List[Dict]:
        """Process documents in batches."""
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = self.pipeline.batch_summarize(batch, method=method)
            results.extend(batch_results)
            
            print(f"Processed {min(i + batch_size, len(documents))}/{len(documents)} documents")
        
        return results
    
    def evaluate_batch(self, documents: List[str], generated_summaries: List[str], 
                      reference_summaries: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate batch of summaries."""
        if reference_summaries:
            return self.pipeline.evaluator.evaluate_batch(reference_summaries, generated_summaries, documents)
        else:
            # Calculate average self-evaluation metrics
            individual_scores = []
            for doc, summ in zip(documents, generated_summaries):
                scores = self.pipeline.evaluate_summary(doc, summ)
                individual_scores.append(scores)
            
            # Average the scores
            avg_scores = {}
            if individual_scores:
                for key in individual_scores[0].keys():
                    avg_scores[key] = sum(score[key] for score in individual_scores) / len(individual_scores)
            
            return avg_scores