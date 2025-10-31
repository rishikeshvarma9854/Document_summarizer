"""
Evaluation metrics for text summarization.
Implements ROUGE scores, BERT-Score, and custom evaluation metrics.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re


class ROUGEEvaluator:
    """ROUGE evaluation metrics for summarization."""
    
    def __init__(self):
        self.stemmer = None
        try:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
        except ImportError:
            pass
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i+n]))
        return Counter(ngrams)
    
    def rouge_n(self, reference: str, candidate: str, n: int = 1) -> Dict[str, float]:
        """Calculate ROUGE-N scores."""
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        
        if len(ref_ngrams) == 0 or len(cand_ngrams) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate overlap
        overlap = sum((ref_ngrams & cand_ngrams).values())
        
        # Calculate precision, recall, F1
        precision = overlap / sum(cand_ngrams.values()) if sum(cand_ngrams.values()) > 0 else 0.0
        recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def rouge_l(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE-L scores using LCS."""
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Calculate LCS length
        lcs_length = self._lcs_length(ref_tokens, cand_tokens)
        
        # Calculate precision, recall, F1
        precision = lcs_length / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def evaluate(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Evaluate multiple reference-candidate pairs."""
        if len(references) != len(candidates):
            raise ValueError("Number of references and candidates must match")
        
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for ref, cand in zip(references, candidates):
            rouge_1_scores.append(self.rouge_n(ref, cand, 1))
            rouge_2_scores.append(self.rouge_n(ref, cand, 2))
            rouge_l_scores.append(self.rouge_l(ref, cand))
        
        # Average scores
        results = {}
        for metric, scores in [('rouge_1', rouge_1_scores), ('rouge_2', rouge_2_scores), ('rouge_l', rouge_l_scores)]:
            results[f'{metric}_precision'] = np.mean([s['precision'] for s in scores])
            results[f'{metric}_recall'] = np.mean([s['recall'] for s in scores])
            results[f'{metric}_f1'] = np.mean([s['f1'] for s in scores])
        
        return results


class BERTScoreEvaluator:
    """BERT-Score evaluation for semantic similarity."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.bert_score = None
        
        try:
            from bert_score import score
            self.bert_score = score
        except ImportError:
            print("bert-score not installed. Install with: pip install bert-score")
    
    def evaluate(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Calculate BERT-Score for reference-candidate pairs."""
        if self.bert_score is None:
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
        
        P, R, F1 = self.bert_score(candidates, references, model_type=self.model_name)
        
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }


class CustomEvaluator:
    """Custom evaluation metrics for summarization."""
    
    def __init__(self):
        pass
    
    def compression_ratio(self, original: str, summary: str) -> float:
        """Calculate compression ratio."""
        orig_words = len(original.split())
        summ_words = len(summary.split())
        return summ_words / orig_words if orig_words > 0 else 0.0
    
    def coverage_score(self, original: str, summary: str) -> float:
        """Calculate how much of the original content is covered."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([original, summary])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarity[0][0]
        except:
            return 0.0
    
    def novelty_score(self, original: str, summary: str) -> float:
        """Calculate novelty (how much new information is in summary)."""
        orig_words = set(original.lower().split())
        summ_words = set(summary.lower().split())
        
        if len(summ_words) == 0:
            return 0.0
        
        novel_words = summ_words - orig_words
        return len(novel_words) / len(summ_words)
    
    def readability_score(self, text: str) -> float:
        """Simple readability score based on sentence and word length."""
        sentences = text.split('.')
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula (lower is better)
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 4.7)
        return max(0, min(100, readability)) / 100  # Normalize to 0-1


class SummarizationEvaluator:
    """Comprehensive evaluation suite for summarization."""
    
    def __init__(self, use_bert_score: bool = True):
        self.rouge_evaluator = ROUGEEvaluator()
        self.custom_evaluator = CustomEvaluator()
        
        if use_bert_score:
            self.bert_evaluator = BERTScoreEvaluator()
        else:
            self.bert_evaluator = None
    
    def evaluate_single(self, reference: str, candidate: str, original: Optional[str] = None) -> Dict[str, float]:
        """Evaluate a single reference-candidate pair."""
        results = {}
        
        # ROUGE scores
        rouge_1 = self.rouge_evaluator.rouge_n(reference, candidate, 1)
        rouge_2 = self.rouge_evaluator.rouge_n(reference, candidate, 2)
        rouge_l = self.rouge_evaluator.rouge_l(reference, candidate)
        
        results.update({
            'rouge_1_f1': rouge_1['f1'],
            'rouge_2_f1': rouge_2['f1'],
            'rouge_l_f1': rouge_l['f1']
        })
        
        # BERT-Score
        if self.bert_evaluator:
            bert_scores = self.bert_evaluator.evaluate([reference], [candidate])
            results.update(bert_scores)
        
        # Custom metrics
        if original:
            results.update({
                'compression_ratio': self.custom_evaluator.compression_ratio(original, candidate),
                'coverage_score': self.custom_evaluator.coverage_score(original, candidate),
                'novelty_score': self.custom_evaluator.novelty_score(original, candidate)
            })
        
        results['readability_score'] = self.custom_evaluator.readability_score(candidate)
        
        return results
    
    def evaluate_batch(self, references: List[str], candidates: List[str], 
                      originals: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate multiple reference-candidate pairs."""
        if len(references) != len(candidates):
            raise ValueError("Number of references and candidates must match")
        
        # ROUGE scores
        rouge_results = self.rouge_evaluator.evaluate(references, candidates)
        
        # BERT-Score
        bert_results = {}
        if self.bert_evaluator:
            bert_results = self.bert_evaluator.evaluate(references, candidates)
        
        # Custom metrics
        custom_results = {}
        if originals and len(originals) == len(candidates):
            compression_ratios = [self.custom_evaluator.compression_ratio(orig, cand) 
                                for orig, cand in zip(originals, candidates)]
            coverage_scores = [self.custom_evaluator.coverage_score(orig, cand) 
                             for orig, cand in zip(originals, candidates)]
            novelty_scores = [self.custom_evaluator.novelty_score(orig, cand) 
                            for orig, cand in zip(originals, candidates)]
            
            custom_results.update({
                'compression_ratio': np.mean(compression_ratios),
                'coverage_score': np.mean(coverage_scores),
                'novelty_score': np.mean(novelty_scores)
            })
        
        readability_scores = [self.custom_evaluator.readability_score(cand) for cand in candidates]
        custom_results['readability_score'] = np.mean(readability_scores)
        
        # Combine all results
        results = {**rouge_results, **bert_results, **custom_results}
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, float]):
        """Print a formatted evaluation report."""
        print("=" * 50)
        print("SUMMARIZATION EVALUATION REPORT")
        print("=" * 50)
        
        # ROUGE scores
        print("\nROUGE Scores:")
        print(f"  ROUGE-1 F1: {results.get('rouge_1_f1', 0):.4f}")
        print(f"  ROUGE-2 F1: {results.get('rouge_2_f1', 0):.4f}")
        print(f"  ROUGE-L F1: {results.get('rouge_l_f1', 0):.4f}")
        
        # BERT-Score
        if 'bert_f1' in results:
            print(f"\nBERT-Score F1: {results['bert_f1']:.4f}")
        
        # Custom metrics
        print("\nCustom Metrics:")
        if 'compression_ratio' in results:
            print(f"  Compression Ratio: {results['compression_ratio']:.4f}")
        if 'coverage_score' in results:
            print(f"  Coverage Score: {results['coverage_score']:.4f}")
        if 'novelty_score' in results:
            print(f"  Novelty Score: {results['novelty_score']:.4f}")
        print(f"  Readability Score: {results.get('readability_score', 0):.4f}")
        
        print("=" * 50)