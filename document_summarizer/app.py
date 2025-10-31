"""
Advanced Document Summarizer with Dark/Light Theme and Enhanced Text Cleaning
"""

import streamlit as st
import re
import numpy as np
from collections import Counter
import io
from datetime import datetime
import nltk

# Download required NLTK data for cloud deployment
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

# Download NLTK data on startup
download_nltk_data()

# File processing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

class AdvancedTextCleaner:
    """Advanced text cleaning to remove headers, footers, and unwanted content."""
    
    def __init__(self):
        # Enhanced patterns for headers and footers
        self.header_patterns = [
            r'KESHAV MEMORIAL INSTITU?TE OF TECHNOLOGY.*?INSTITUTION\)',
            r'WEB TECHNOLOGIES.*?PAGE NO:\d+',
            r'UNIT\s*-\s*[IVX]+.*?REGULATIONS',
            r'KR\d+\s+Regulations.*?INSTITUTION\)',
            r'KMIT.*?AUTONOMOUS INSTITUTION.*?PAGE NO:\d+',
            r'AN AUTONOMO?US INSTITUTION.*?(?=\n)',
            r'PAGE NO:\d+.*?(?=\n)',
            r'Syllabus:.*?(?=\n\n|\n[A-Z])',
            r'Figure:.*?(?=\n)',
            r'Example.*?:.*?(?=\n\n)',
            r'Filename:.*?(?=\n)',
            r'Output.*?:.*?(?=\n\n)',
            r'Browser.*?output.*?:.*?(?=\n\n)',
            r'Console.*?output.*?:.*?(?=\n\n)',
            r'Terminal.*?output.*?:.*?(?=\n\n)',
        ]
        
        # Enhanced repetitive patterns
        self.repetitive_patterns = [
            r'KESHAV MEMORIAL.*?(?=\n)',
            r'WEB TECHNOLOGIES.*?(?=\n)',
            r'UNIT\s*-\s*[IVX]+.*?(?=\n)',
            r'KR\d+.*?(?=\n)',
            r'KMIT.*?(?=\n)',
            r'AN AUTONOMO?US.*?(?=\n)',
            r'INSTITUTION.*?(?=\n)',
            r'PAGE NO:\d+.*?(?=\n)',
            r'Run:.*?js.*?(?=\n)',
            r'node.*?js.*?(?=\n)',
            r'npm.*?(?=\n)',
            r'Browser.*?url:.*?(?=\n)',
            r'localhost:\d+.*?(?=\n)',
            r'Question:.*?(?=\n)',
            r'Query:.*?(?=\n)',
            r'Write a query.*?(?=\n)',
            r'Q-?\d+\..*?(?=\n)',
        ]
        
        # Patterns for code blocks and technical jargon
        self.code_patterns = [
            r'const\s+\w+.*?;',
            r'require\s*\(.*?\)',
            r'console\.log\s*\(.*?\)',
            r'app\.\w+\s*\(.*?\)',
            r'res\.\w+\s*\(.*?\)',
            r'req\.\w+.*?(?=\s)',
            r'db\.\w+.*?(?=\s)',
            r'mongod.*?(?=\s)',
            r'mongosh.*?(?=\s)',
        ]
    
    def clean_academic_content(self, text):
        """Remove academic headers, footers, and repetitive content."""
        if not isinstance(text, str):
            return ""
        
        # Remove header patterns (more aggressive)
        for pattern in self.header_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        # Remove repetitive patterns (more aggressive)
        for pattern in self.repetitive_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove specific institutional references
        institutional_patterns = [
            r'KESHAV MEMORIAL.*?TECHNOLOGY',
            r'AN AUTONOMOUS INSTITUTION',
            r'INSTITU?TE OF TECHNOLOGY',
            r'\(AN AUTONOMO?US INSTITUTION\)',
        ]
        
        for pattern in institutional_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        
        # Remove standalone numbers and page references
        text = re.sub(r'\b\d+\s*\)', '', text)  # Remove numbered lists like "1)"
        text = re.sub(r'\bpage\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'PAGE NO:\d+', '', text, flags=re.IGNORECASE)
        
        # Remove URLs and email patterns
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation and special characters
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        
        # Remove very short fragments
        sentences = text.split('.')
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
        text = '. '.join(meaningful_sentences)
        
        # Clean up and return
        return text.strip()
    
    def extract_meaningful_content(self, text):
        """Extract only meaningful sentences and paragraphs."""
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short sentences
            if len(sentence.split()) < 5:
                continue
            
            # Skip sentences that are mostly technical jargon
            if self._is_mostly_technical(sentence):
                continue
            
            # Skip sentences with too many special characters
            if len(re.findall(r'[^\w\s]', sentence)) > len(sentence.split()) * 0.3:
                continue
            
            meaningful_sentences.append(sentence)
        
        return '. '.join(meaningful_sentences)
    
    def _is_mostly_technical(self, sentence):
        """Check if sentence is mostly technical code or jargon."""
        technical_indicators = [
            'const', 'require', 'console', 'app.', 'res.', 'req.',
            'function', 'var', 'let', 'npm', 'node', 'js',
            'localhost', 'port', 'server', 'http'
        ]
        
        words = sentence.lower().split()
        technical_count = sum(1 for word in words if any(tech in word for tech in technical_indicators))
        
        return technical_count > len(words) * 0.4

class DocumentProcessor:
    """Process different document formats with advanced cleaning."""
    
    def __init__(self):
        self.text_cleaner = AdvancedTextCleaner()
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract and clean text from PDF file."""
        if not PDF_AVAILABLE:
            return None, "PyPDF2 not installed. Install with: pip install PyPDF2"
        
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Apply advanced cleaning
            cleaned_text = self.text_cleaner.clean_academic_content(text)
            meaningful_text = self.text_cleaner.extract_meaningful_content(cleaned_text)
            
            return meaningful_text.strip(), None
        except Exception as e:
            return None, f"Error reading PDF: {str(e)}"
    
    def extract_text_from_docx(self, docx_file):
        """Extract and clean text from DOCX file."""
        if not DOCX_AVAILABLE:
            return None, "python-docx not installed. Install with: pip install python-docx"
        
        try:
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Apply advanced cleaning
            cleaned_text = self.text_cleaner.clean_academic_content(text)
            meaningful_text = self.text_cleaner.extract_meaningful_content(cleaned_text)
            
            return meaningful_text.strip(), None
        except Exception as e:
            return None, f"Error reading DOCX: {str(e)}"
    
    def extract_text_from_txt(self, txt_file):
        """Extract and clean text from TXT file."""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    txt_file.seek(0)
                    text = txt_file.read().decode(encoding)
                    
                    # Apply advanced cleaning
                    cleaned_text = self.text_cleaner.clean_academic_content(text)
                    meaningful_text = self.text_cleaner.extract_meaningful_content(cleaned_text)
                    
                    return meaningful_text.strip(), None
                except UnicodeDecodeError:
                    continue
            return None, "Could not decode text file. Please ensure it's in UTF-8 format."
        except Exception as e:
            return None, f"Error reading TXT: {str(e)}"
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded file and extract clean text."""
        if uploaded_file is None:
            return None, "No file uploaded"
        
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            return self.extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(uploaded_file)
        else:
            return None, f"Unsupported file format: {file_extension}"

class SummaryFormatter:
    """Format summaries with proper structure and headings."""
    
    def format_summary(self, summary_text, original_filename):
        """Format summary with headings and structure."""
        sentences = [s.strip() for s in summary_text.split('.') if s.strip() and len(s.strip()) > 10]
        
        if len(sentences) <= 2:
            return f"# Summary of {original_filename}\n\n{summary_text}"
        
        # Group sentences into sections
        formatted_summary = f"# Summary of {original_filename}\n\n"
        
        if len(sentences) >= 4:
            # Add overview section
            formatted_summary += "## Overview\n\n"
            formatted_summary += f"{sentences[0]}.\n\n"
            
            # Add key points section
            formatted_summary += "## Key Points\n\n"
            
            # Take middle sentences as key points
            key_points = sentences[1:min(len(sentences)-1, 4)]
            for i, point in enumerate(key_points, 1):
                formatted_summary += f"**{i}.** {point.strip()}.\n\n"
            
            # Add conclusion if there are enough sentences
            if len(sentences) > 4:
                formatted_summary += "## Additional Details\n\n"
                remaining_sentences = sentences[4:]
                for sentence in remaining_sentences:
                    formatted_summary += f"• {sentence.strip()}.\n\n"
                    
            # Add conclusion
            if len(sentences) > 1:
                formatted_summary += "## Conclusion\n\n"
                formatted_summary += f"{sentences[-1]}.\n\n"
        else:
            # Simple format for short summaries
            formatted_summary += "## Key Information\n\n"
            for i, sentence in enumerate(sentences, 1):
                formatted_summary += f"**{i}.** {sentence.strip()}.\n\n"
        
        return formatted_summary.strip()

class SmartTextRankSummarizer:
    """Enhanced TextRank implementation with better sentence selection."""
    
    def __init__(self):
        self.formatter = SummaryFormatter()
    
    def segment_sentences(self, text):
        """Split text into meaningful sentences."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 8]
        return sentences
    
    def _tokenize(self, text):
        """Advanced tokenization with stopword filtering."""
        import re
        tokens = re.findall(r'\w+', text.lower())
        
        # Basic stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        return [token for token in tokens if token not in stopwords and len(token) > 2]
    
    def _compute_tf_idf(self, sentences):
        """Compute TF-IDF vectors for sentences."""
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
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def _build_similarity_matrix(self, sentences):
        """Build similarity matrix between sentences."""
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
    
    def summarize(self, text, compression_ratio=0.3, filename="document"):
        """Generate smart extractive summary using TextRank."""
        sentences = self.segment_sentences(text)
        
        if len(sentences) <= 3:
            raw_summary = ' '.join(sentences)
        else:
            # Calculate number of sentences based on compression ratio (more generous)
            base_sentences = max(5, int(len(sentences) * compression_ratio))
            
            # Ensure we get a good number of sentences for comprehensive summary
            if compression_ratio <= 0.2:
                num_sentences = max(4, base_sentences)  # At least 4 sentences for very short
            elif compression_ratio <= 0.4:
                num_sentences = max(6, base_sentences)  # At least 6 sentences for balanced
            else:
                num_sentences = max(8, base_sentences)  # At least 8 sentences for detailed
            
            num_sentences = min(num_sentences, len(sentences))
            
            try:
                # Build similarity matrix and compute scores
                similarity_matrix = self._build_similarity_matrix(sentences)
                scores = self._textrank_scores(similarity_matrix)
                
                # Get top sentences with better distribution
                top_indices = np.argsort(scores)[-num_sentences:][::-1]
                
                # Ensure we include first and last sentences for context
                if 0 not in top_indices and len(sentences) > 3:
                    top_indices = np.append(top_indices[:-1], 0)
                if (len(sentences)-1) not in top_indices and len(sentences) > 5:
                    top_indices = np.append(top_indices[:-1], len(sentences)-1)
                
                top_indices = sorted(top_indices)  # Maintain original order
                
                summary_sentences = [sentences[i] for i in top_indices]
                raw_summary = '. '.join(summary_sentences) + '.'
            except Exception as e:
                st.error(f"TextRank failed: {e}")
                # Better fallback - distribute sentences across the document
                step = max(1, len(sentences) // num_sentences)
                selected_indices = list(range(0, len(sentences), step))[:num_sentences]
                raw_summary = '. '.join([sentences[i] for i in selected_indices]) + '.'
        
        # Format the summary with proper structure
        formatted_summary = self.formatter.format_summary(raw_summary, filename)
        return formatted_summary

@st.cache_resource
def load_components():
    """Load components with caching."""
    try:
        doc_processor = DocumentProcessor()
        summarizer = SmartTextRankSummarizer()
        return doc_processor, summarizer
    except Exception as e:
        st.error(f"Error loading components: {e}")
        # Fallback without caching
        return DocumentProcessor(), SmartTextRankSummarizer()

def apply_custom_css():
    """Apply custom CSS for better UI."""
    st.markdown("""
    <style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Style the file uploader container */
    .stFileUploader {
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stFileUploader:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* File uploader text styling */
    .stFileUploader label {
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Summary box styling */
    .summary-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    /* Improved button styling */
    .stButton > button {
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Upload area styling */
    .upload-area {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def create_download_buttons(text, base_filename):
    """Create download buttons for different formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # TXT download
        txt_filename = f"{base_filename}_summary_{timestamp}.txt"
        txt_buffer = io.BytesIO()
        txt_buffer.write(text.encode('utf-8'))
        txt_buffer.seek(0)
        
        st.download_button(
            label="📄 Download TXT",
            data=txt_buffer.getvalue(),
            file_name=txt_filename,
            mime="text/plain",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        # PDF download (using reportlab if available, otherwise HTML to PDF)
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            
            pdf_filename = f"{base_filename}_summary_{timestamp}.pdf"
            pdf_buffer = io.BytesIO()
            
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Add title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
            )
            story.append(Paragraph(f"Summary of {base_filename}", title_style))
            story.append(Spacer(1, 12))
            
            # Add content
            for line in text.split('\n'):
                if line.strip():
                    if line.startswith('#'):
                        # Header
                        story.append(Paragraph(line.replace('#', '').strip(), styles['Heading2']))
                    elif line.startswith('•') or line.startswith('-'):
                        # Bullet point
                        story.append(Paragraph(line, styles['Normal']))
                    else:
                        # Regular text
                        story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            pdf_buffer.seek(0)
            
            st.download_button(
                label="📕 Download PDF",
                data=pdf_buffer.getvalue(),
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        except ImportError:
            st.button("📕 PDF (Install reportlab)", disabled=True, use_container_width=True)
    
    with col3:
        # DOCX download
        try:
            from docx import Document
            from docx.shared import Inches
            
            docx_filename = f"{base_filename}_summary_{timestamp}.docx"
            docx_buffer = io.BytesIO()
            
            doc = Document()
            doc.add_heading(f'Summary of {base_filename}', 0)
            
            for line in text.split('\n'):
                if line.strip():
                    if line.startswith('# '):
                        doc.add_heading(line.replace('#', '').strip(), level=1)
                    elif line.startswith('## '):
                        doc.add_heading(line.replace('##', '').strip(), level=2)
                    elif line.startswith('•') or line.startswith('-'):
                        p = doc.add_paragraph(line.replace('•', '').replace('-', '').strip())
                        p.style = 'List Bullet'
                    elif line.strip() and not line.startswith('#'):
                        doc.add_paragraph(line.strip())
            
            doc.save(docx_buffer)
            docx_buffer.seek(0)
            
            st.download_button(
                label="📘 Download DOCX",
                data=docx_buffer.getvalue(),
                file_name=docx_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True
            )
        except ImportError:
            st.button("📘 DOCX (Available)", disabled=True, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Smart Document Summarizer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Hide sidebar completely
    st.markdown("""
    <style>
    .css-1d391kg {display: none}
    .css-1rs6os {display: none}
    .css-17eq0hr {display: none}
    section[data-testid="stSidebar"] {display: none}
    </style>
    """, unsafe_allow_html=True)
    
    # Apply custom CSS
    apply_custom_css()
    

    

    
    # Apply clean theme styles
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    .stFileUploader {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%) !important;
        border: 2px dashed rgba(59, 130, 246, 0.5) !important;
    }
    .stFileUploader:hover {
        border-color: rgba(59, 130, 246, 0.8) !important;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(147, 51, 234, 0.15) 100%) !important;
    }
    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.8) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    .summary-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(51, 65, 85, 0.6) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
    }
    .metric-container {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Smart Document Summarizer</h1>
        <p>Upload your documents and get AI-powered summaries with advanced text cleaning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load components (try cached first, fallback to direct)
    try:
        doc_processor, summarizer = load_components()
    except Exception as e:
        st.warning(f"Using fallback components due to: {e}")
        doc_processor = DocumentProcessor()
        summarizer = SmartTextRankSummarizer()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📁 Upload Your Document")
        
        uploaded_file = st.file_uploader(
            "📄 Drag & Drop Your File Here or Click to Browse\n\nSupports PDF, DOCX, and TXT files",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        # Compression ratio slider
        st.markdown("### ⚙️ Summary Settings")
        compression_ratio = st.slider(
            "Summary Length",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.1,
            help="Lower values = shorter summaries"
        )
        
        # Display compression info
        if compression_ratio <= 0.2:
            st.info("📝 Very Short Summary (Key points only)")
        elif compression_ratio <= 0.4:
            st.info("📄 Balanced Summary (Main ideas)")
        else:
            st.info("📚 Detailed Summary (Comprehensive)")
    
    with col2:
        st.markdown("### 📊 Document Analysis")
        
        if uploaded_file is not None:
            # Display file info with nice styling
            st.markdown(f"""
            <div class="metric-container">
                <h4>📄 {uploaded_file.name}</h4>
                <p>Size: {uploaded_file.size:,} bytes</p>
                <p>Type: {uploaded_file.type}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Process the uploaded file
            with st.spinner("🔍 Extracting and cleaning text..."):
                extracted_text, error = doc_processor.process_uploaded_file(uploaded_file)
            
            if error:
                st.error(f"❌ {error}")
                
                if "not installed" in error:
                    st.markdown("""
                    **📦 Install required packages:**
                    ```bash
                    pip install PyPDF2 python-docx
                    ```
                    """)
            
            elif extracted_text:
                # Show extracted text info
                word_count = len(extracted_text.split())
                char_count = len(extracted_text)
                
                # Metrics in a nice layout
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("📊 Words", f"{word_count:,}")
                with metric_col2:
                    st.metric("📝 Characters", f"{char_count:,}")
                
                # Show preview of cleaned text
                with st.expander("👀 Preview Cleaned Text"):
                    preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                    st.text_area("Clean Document Content", preview_text, height=200)
                
                # Generate summary button
                if st.button("🚀 Generate Smart Summary", type="primary", use_container_width=True):
                    with st.spinner("🤖 Creating intelligent summary..."):
                        try:
                            original_name = uploaded_file.name.rsplit('.', 1)[0]
                            
                            # Debug info
                            st.write(f"Debug: Summarizer type: {type(summarizer)}")
                            st.write(f"Debug: Method signature: {summarizer.summarize.__code__.co_varnames}")
                            
                            summary = summarizer.summarize(
                                text=extracted_text, 
                                compression_ratio=compression_ratio, 
                                filename=original_name
                            )
                        except Exception as e:
                            st.error(f"Error during summarization: {e}")
                            st.error(f"Error type: {type(e)}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                            summary = None
                    
                    if summary:
                        st.success("✅ Summary generated successfully!")
                        
                        # Display formatted summary
                        st.markdown("### 📋 Smart Summary")
                        st.markdown("---")
                        st.markdown(summary)
                        st.markdown("---")
                        
                        # Summary statistics
                        summary_words = len(summary.split())
                        actual_compression = summary_words / word_count if word_count > 0 else 0
                        
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("📊 Summary Words", f"{summary_words:,}")
                        with stat_col2:
                            st.metric("📉 Compression", f"{actual_compression:.2f}")
                        with stat_col3:
                            st.metric("⚡ Reduction", f"{(1-actual_compression)*100:.1f}%")
                        
                        # Download buttons for multiple formats
                        st.markdown("### 📥 Download Summary")
                        create_download_buttons(summary, original_name)
                        
                    else:
                        st.error("❌ Failed to generate summary. Please try with a different document.")
            
            else:
                st.warning("⚠️ No meaningful text could be extracted from the document.")
        
        else:
            # Show instructions when no file is uploaded
            st.markdown("""
            <div class="summary-box">
                <h3>🚀 How to Use</h3>
                <p><strong>1.</strong> Upload your document (PDF, DOCX, or TXT)</p>
                <p><strong>2.</strong> Adjust the summary length slider</p>
                <p><strong>3.</strong> Click "Generate Smart Summary"</p>
                <p><strong>4.</strong> Download your clean, intelligent summary</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature highlights
            st.markdown("### ✨ Features")
            st.markdown("""
            - 🧹 **Advanced Text Cleaning**: Removes headers, footers, and academic jargon
            - 🤖 **Smart Summarization**: TextRank algorithm with enhanced sentence selection
            - 🎨 **Beautiful UI**: Dark/Light theme with modern design
            - 📥 **Easy Download**: Get summaries as timestamped text files
            - 🔒 **Privacy First**: All processing happens locally
            """)
    

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; backdrop-filter: blur(10px);'>
        <p>🤖 <strong>Smart Document Summarizer</strong> | Powered by Advanced TextRank Algorithm</p>
        <p>🔒 Your documents are processed locally - no data is stored or shared</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()