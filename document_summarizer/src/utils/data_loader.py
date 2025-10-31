"""
Data loading utilities for summarization datasets.
Handles various dataset formats and preprocessing pipelines.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple, Iterator
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from .preprocessing import TextPreprocessor, DocumentProcessor


class SummarizationDataset(Dataset):
    """PyTorch Dataset for summarization tasks."""
    
    def __init__(self, documents: List[str], summaries: Optional[List[str]] = None,
                 preprocessor: Optional[TextPreprocessor] = None, max_length: int = 512):
        self.documents = documents
        self.summaries = summaries
        self.preprocessor = preprocessor or TextPreprocessor()
        self.max_length = max_length
        
        # Process documents
        self.processed_data = []
        for i, doc in enumerate(documents):
            processed = self.preprocessor.clean_text(doc)
            sentences = self.preprocessor.segment_sentences(processed)
            
            item = {
                'document': processed,
                'sentences': sentences,
                'num_sentences': len(sentences)
            }
            
            if summaries and i < len(summaries):
                item['summary'] = self.preprocessor.clean_text(summaries[i])
            
            self.processed_data.append(item)
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.processed_data[idx]


class DatasetLoader:
    """Load and manage various summarization datasets."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = TextPreprocessor()
    
    def load_cnn_dailymail(self, split: str = "train", max_samples: Optional[int] = None) -> SummarizationDataset:
        """Load CNN/DailyMail dataset."""
        try:
            from datasets import load_dataset
            
            cache_file = self.cache_dir / f"cnn_dailymail_{split}_{max_samples or 'all'}.json"
            
            if cache_file.exists():
                print(f"Loading cached data from {cache_file}")
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                documents = data['documents']
                summaries = data['summaries']
            else:
                print(f"Downloading CNN/DailyMail {split} split...")
                dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
                
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                
                documents = [item['article'] for item in dataset]
                summaries = [item['highlights'] for item in dataset]
                
                # Cache the data
                with open(cache_file, 'w') as f:
                    json.dump({'documents': documents, 'summaries': summaries}, f)
            
            return SummarizationDataset(documents, summaries, self.preprocessor)
            
        except ImportError:
            raise ImportError("datasets library not installed. Install with: pip install datasets")
    
    def load_xsum(self, split: str = "train", max_samples: Optional[int] = None) -> SummarizationDataset:
        """Load XSum dataset."""
        try:
            from datasets import load_dataset
            
            cache_file = self.cache_dir / f"xsum_{split}_{max_samples or 'all'}.json"
            
            if cache_file.exists():
                print(f"Loading cached data from {cache_file}")
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                documents = data['documents']
                summaries = data['summaries']
            else:
                print(f"Downloading XSum {split} split...")
                dataset = load_dataset("xsum", split=split)
                
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                
                documents = [item['document'] for item in dataset]
                summaries = [item['summary'] for item in dataset]
                
                # Cache the data
                with open(cache_file, 'w') as f:
                    json.dump({'documents': documents, 'summaries': summaries}, f)
            
            return SummarizationDataset(documents, summaries, self.preprocessor)
            
        except ImportError:
            raise ImportError("datasets library not installed. Install with: pip install datasets")
    
    def load_reddit_tifu(self, split: str = "train", max_samples: Optional[int] = None) -> SummarizationDataset:
        """Load Reddit TIFU dataset."""
        try:
            from datasets import load_dataset
            
            cache_file = self.cache_dir / f"reddit_tifu_{split}_{max_samples or 'all'}.json"
            
            if cache_file.exists():
                print(f"Loading cached data from {cache_file}")
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                documents = data['documents']
                summaries = data['summaries']
            else:
                print(f"Downloading Reddit TIFU {split} split...")
                dataset = load_dataset("reddit_tifu", "long", split=split)
                
                if max_samples:
                    dataset = dataset.select(range(min(max_samples, len(dataset))))
                
                documents = [item['documents'] for item in dataset]
                summaries = [item['tldr'] for item in dataset]
                
                # Cache the data
                with open(cache_file, 'w') as f:
                    json.dump({'documents': documents, 'summaries': summaries}, f)
            
            return SummarizationDataset(documents, summaries, self.preprocessor)
            
        except ImportError:
            raise ImportError("datasets library not installed. Install with: pip install datasets")
    
    def load_custom_dataset(self, file_path: str, text_column: str = "text", 
                           summary_column: Optional[str] = "summary") -> SummarizationDataset:
        """Load custom dataset from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix == '.jsonl':
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        documents = df[text_column].tolist()
        summaries = df[summary_column].tolist() if summary_column and summary_column in df.columns else None
        
        return SummarizationDataset(documents, summaries, self.preprocessor)
    
    def create_dataloader(self, dataset: SummarizationDataset, batch_size: int = 8, 
                         shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """Create PyTorch DataLoader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function for DataLoader."""
        documents = [item['document'] for item in batch]
        sentences_list = [item['sentences'] for item in batch]
        num_sentences = [item['num_sentences'] for item in batch]
        
        result = {
            'documents': documents,
            'sentences': sentences_list,
            'num_sentences': num_sentences
        }
        
        if 'summary' in batch[0]:
            summaries = [item['summary'] for item in batch]
            result['summaries'] = summaries
        
        return result


class DocumentSampler:
    """Sample documents for evaluation and testing."""
    
    def __init__(self, min_length: int = 100, max_length: int = 2000):
        self.min_length = min_length
        self.max_length = max_length
        self.preprocessor = TextPreprocessor()
    
    def filter_by_length(self, documents: List[str], summaries: Optional[List[str]] = None) -> Tuple[List[str], Optional[List[str]]]:
        """Filter documents by length criteria."""
        filtered_docs = []
        filtered_summaries = [] if summaries else None
        
        for i, doc in enumerate(documents):
            word_count = len(doc.split())
            if self.min_length <= word_count <= self.max_length:
                filtered_docs.append(doc)
                if summaries and i < len(summaries):
                    filtered_summaries.append(summaries[i])
        
        return filtered_docs, filtered_summaries
    
    def sample_by_domain(self, documents: List[str], summaries: Optional[List[str]] = None, 
                        domain_keywords: Dict[str, List[str]] = None) -> Dict[str, Tuple[List[str], Optional[List[str]]]]:
        """Sample documents by domain based on keywords."""
        if domain_keywords is None:
            domain_keywords = {
                'news': ['news', 'report', 'journalist', 'breaking', 'update'],
                'science': ['research', 'study', 'experiment', 'analysis', 'findings'],
                'business': ['company', 'market', 'financial', 'revenue', 'profit'],
                'technology': ['software', 'algorithm', 'computer', 'digital', 'tech']
            }
        
        domain_samples = {domain: ([], []) for domain in domain_keywords}
        
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            
            for domain, keywords in domain_keywords.items():
                if any(keyword in doc_lower for keyword in keywords):
                    domain_samples[domain][0].append(doc)
                    if summaries and i < len(summaries):
                        domain_samples[domain][1].append(summaries[i])
                    break
        
        # Convert to proper format
        result = {}
        for domain, (docs, summs) in domain_samples.items():
            result[domain] = (docs, summs if summaries else None)
        
        return result
    
    def create_evaluation_split(self, documents: List[str], summaries: Optional[List[str]] = None,
                               train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, Tuple[List[str], Optional[List[str]]]]:
        """Create train/validation/test splits."""
        n = len(documents)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': (documents[:train_end], summaries[:train_end] if summaries else None),
            'validation': (documents[train_end:val_end], summaries[train_end:val_end] if summaries else None),
            'test': (documents[val_end:], summaries[val_end:] if summaries else None)
        }
        
        return splits


class DatasetStatistics:
    """Calculate and display dataset statistics."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def calculate_stats(self, documents: List[str], summaries: Optional[List[str]] = None) -> Dict:
        """Calculate comprehensive dataset statistics."""
        doc_lengths = [len(doc.split()) for doc in documents]
        doc_sentences = [len(self.preprocessor.segment_sentences(doc)) for doc in documents]
        
        stats = {
            'num_documents': len(documents),
            'avg_doc_length': sum(doc_lengths) / len(doc_lengths),
            'min_doc_length': min(doc_lengths),
            'max_doc_length': max(doc_lengths),
            'avg_sentences_per_doc': sum(doc_sentences) / len(doc_sentences),
            'total_words': sum(doc_lengths)
        }
        
        if summaries:
            summ_lengths = [len(summ.split()) for summ in summaries]
            compression_ratios = [summ_len / doc_len if doc_len > 0 else 0 
                                for summ_len, doc_len in zip(summ_lengths, doc_lengths)]
            
            stats.update({
                'avg_summary_length': sum(summ_lengths) / len(summ_lengths),
                'min_summary_length': min(summ_lengths),
                'max_summary_length': max(summ_lengths),
                'avg_compression_ratio': sum(compression_ratios) / len(compression_ratios),
                'total_summary_words': sum(summ_lengths)
            })
        
        return stats
    
    def print_stats(self, stats: Dict):
        """Print formatted statistics."""
        print("=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        
        print(f"Number of documents: {stats['num_documents']:,}")
        print(f"Total words: {stats['total_words']:,}")
        print(f"Average document length: {stats['avg_doc_length']:.1f} words")
        print(f"Document length range: {stats['min_doc_length']} - {stats['max_doc_length']} words")
        print(f"Average sentences per document: {stats['avg_sentences_per_doc']:.1f}")
        
        if 'avg_summary_length' in stats:
            print(f"\nSummary Statistics:")
            print(f"Total summary words: {stats['total_summary_words']:,}")
            print(f"Average summary length: {stats['avg_summary_length']:.1f} words")
            print(f"Summary length range: {stats['min_summary_length']} - {stats['max_summary_length']} words")
            print(f"Average compression ratio: {stats['avg_compression_ratio']:.3f}")
        
        print("=" * 50)