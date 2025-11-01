# 🤖 Smart Document Summarizer

An advanced document summarization system powered by state-of-the-art transformer models (T5, BART, FLAN-T5) with intelligent text processing. Upload files or paste URLs from Google Drive, Google Docs, Dropbox, and other platforms to get AI-powered summaries with professional formatting.

## ✨ Key Features

### 📁 **File Processing**
- **PDF Support**: Extracts text from research papers, reports, articles, and books
- **DOCX Support**: Processes Word documents, essays, proposals, and letters  
- **TXT Support**: Handles plain text files, notes, scripts, and logs
- **URL Support**: Process documents from Google Drive, Google Docs, Dropbox, and direct links
- **Drag & Drop Interface**: Beautiful upload area with visual feedback

### 🧹 **Advanced Text Cleaning**
- **Academic Content Filtering**: Removes institutional headers like "KESHAV MEMORIAL INSTITUTE OF TECHNOLOGY"
- **Header/Footer Removal**: Eliminates page numbers, "PAGE NO:", "AN AUTONOMOUS INSTITUTION"
- **Repetitive Content**: Filters out repeated academic references and formatting
- **Question Pattern Removal**: Removes "Q-1:", "Write a query", "Question:" patterns
- **Code Block Filtering**: Eliminates technical jargon and programming snippets
- **Meaningful Content Extraction**: Keeps only readable, informative sentences

### 🤖 **AI-Powered Summarization with Transformers**
- **T5 Transformer (Default)**: Google's text-to-text transfer transformer for advanced abstractive summarization
- **BART Model**: Facebook's bidirectional auto-regressive transformer for high-quality generation
- **FLAN-T5 (Advanced)**: Google's instruction-tuned T5 model for better summarization quality
- **Hugging Face Integration**: Hardcoded API key for seamless model access
- **TextRank Fallback**: Fast extractive summarization when transformers aren't available
- **Clean Formatting**: Professional paragraph-style output like NotebookLM
- **Adjustable Compression**: AI-optimized summary length from 10%-80% of original

### 📝 **Professional Formatting**
- **Structured Output**: Organized with Overview, Key Points, Additional Details, and Conclusion
- **Markdown Formatting**: Clean headings, bold text, and bullet points
- **Numbered Key Points**: Clear, easy-to-read structure
- **Contextual Sections**: Logical flow from overview to conclusion

### 📥 **Multiple Download Formats**
- **📄 TXT**: Plain text with markdown formatting
- **📕 PDF**: Professional PDF with proper headings and structure
- **📘 DOCX**: Word document with formatted sections and bullet points
- **Timestamped Files**: Automatic naming with date and time

### 🎨 **Beautiful Interface**
- **Dark Theme**: Professional dark blue gradient background
- **Responsive Design**: Works perfectly on desktop and mobile
- **Visual Feedback**: Hover effects and smooth transitions
- **Clean Layout**: No distractions, focused on functionality
- **Real-time Metrics**: Word count, compression ratio, reduction percentage

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)
```bash
python start.py
```
This will automatically:
- Install all required dependencies
- Download necessary NLTK data
- Launch the web interface in your browser

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### 🔗 URL Support
The app now supports processing documents from:
- **Google Drive**: `https://drive.google.com/file/d/...`
- **Google Docs**: `https://docs.google.com/document/d/...`
- **Dropbox**: Direct file links
- **Any direct file URL**: PDF, TXT, DOCX links



## 📊 How It Works

### 1. **Upload Your Document**
- Drag and drop your file or click to browse
- Supports PDF, DOCX, and TXT formats
- Instant file information display

### 2. **Advanced Text Processing**
- Extracts text from your document
- Applies intelligent cleaning algorithms
- Removes headers, footers, and academic jargon
- Filters out repetitive and meaningless content

### 3. **Smart Summarization**
- Uses enhanced TextRank algorithm
- Builds TF-IDF vectors for sentence comparison
- Applies PageRank scoring for importance ranking
- Selects optimal sentences while maintaining context

### 4. **Professional Formatting**
- Structures summary with clear sections
- Adds proper headings and formatting
- Creates logical flow from overview to conclusion
- Ensures readability and professional appearance

### 5. **Multi-Format Export**
- Generate TXT files with markdown formatting
- Create professional PDFs with proper layout
- Export Word documents with structured formatting
- Automatic timestamped filenames

## 📈 Example Results

### Input Document (Academic Paper):
```
KESHAV MEMORIAL INSTITUTE OF TECHNOLOGY (AN AUTONOMOUS INSTITUTION)
WEB TECHNOLOGIES UNIT-III PAGE NO:1

MongoDB is a NoSQL database that uses BSON format for storing documents.
You can use MongoDB Compass for visual database management and querying.
Basic operations include creating collections, inserting documents, and querying data.

KESHAV MEMORIAL INSTITUTE OF TECHNOLOGY (AN AUTONOMOUS INSTITUTION)
PAGE NO:2

Aggregation pipeline allows complex data processing with stages like match, group, and sort.
Import and export functionality helps with data migration and backup.
```

### Generated Summary:
```markdown
# Summary of MongoDB_Tutorial

## Overview
MongoDB is a NoSQL database that uses BSON format for storing documents.

## Key Points
**1.** You can use MongoDB Compass for visual database management and querying.
**2.** Basic operations include creating collections, inserting documents, and querying data.
**3.** Aggregation pipeline allows complex data processing with stages like match, group, and sort.

## Additional Details
• Import and export functionality helps with data migration and backup.

## Conclusion
MongoDB provides flexible document storage and powerful querying capabilities for modern applications.
```

## 🎯 Perfect For

### 👨‍🎓 **Students**
- Summarize research papers and academic articles
- Extract key points from lengthy textbooks
- Clean up messy academic documents
- Create study notes from course materials

### 👨‍💼 **Professionals**
- Quickly digest business reports and proposals
- Extract insights from technical documentation
- Summarize meeting notes and presentations
- Process industry research and whitepapers

### 👨‍🔬 **Researchers**
- Analyze academic papers and journals
- Extract methodology and findings
- Process literature reviews
- Summarize conference proceedings

### 👨‍💻 **Content Creators**
- Summarize source materials for articles
- Extract quotes and key information
- Process interview transcripts
- Condense research for blog posts

## 🔧 Technical Details

### **Core Algorithm**
- **TextRank**: Graph-based ranking algorithm similar to PageRank
- **TF-IDF Vectorization**: Term frequency-inverse document frequency scoring
- **Cosine Similarity**: Measures sentence similarity for graph construction
- **Sentence Filtering**: Removes low-quality and repetitive sentences

### **Text Cleaning Pipeline**
1. **Institutional Header Removal**: Filters academic institution references
2. **Page Number Elimination**: Removes page numbers and navigation elements
3. **Repetitive Content Detection**: Identifies and removes repeated phrases
4. **Question Pattern Filtering**: Removes Q&A formatting and test questions
5. **Technical Jargon Filtering**: Eliminates code snippets and technical commands
6. **Meaningful Content Extraction**: Keeps only informative, readable sentences

### **Formatting Engine**
- **Section Detection**: Automatically creates logical sections
- **Hierarchy Building**: Establishes proper heading structure
- **Content Organization**: Groups related information together
- **Output Generation**: Creates clean, professional formatting

## 📦 Dependencies

### **Core Libraries**
- `streamlit` - Web interface framework
- `numpy` - Numerical computations for algorithms
- `scikit-learn` - TF-IDF vectorization and similarity calculations
- `nltk` - Natural language processing and tokenization

### **File Processing**
- `PyPDF2` - PDF text extraction and processing
- `python-docx` - Word document reading and writing
- `reportlab` - Professional PDF generation

### **Data Handling**
- `pandas` - Data manipulation and analysis
- `collections` - Counter for frequency analysis

## 🔒 Privacy & Security

### **Local Processing**
- ✅ All document processing happens on your machine
- ✅ No files uploaded to external servers
- ✅ No data stored or cached permanently
- ✅ Complete privacy and confidentiality

### **Data Handling**
- ✅ Documents processed in memory only
- ✅ Temporary files automatically cleaned up
- ✅ No logging of document content
- ✅ Secure file handling practices

## 📁 Project Structure

```
document_summarizer/
├── src/                          # Core source code
│   ├── models/                   # AI models and algorithms
│   │   ├── abstractive.py       # T5, BART, Pegasus models
│   │   ├── extractive.py        # BERT, TextRank, clustering
│   │   └── attention.py         # Custom attention mechanisms
│   ├── utils/                   # Utility functions
│   │   ├── preprocessing.py     # Text cleaning and processing
│   │   ├── evaluation.py        # ROUGE and custom metrics
│   │   └── data_loader.py       # Dataset loading utilities
│   └── pipeline.py              # Main processing pipeline
├── app.py                       # 🌟 Main Streamlit application
├── start.py                     # Automatic setup and launch script
├── requirements.txt             # Python dependencies
└── README.md                    # This documentation
```

## 🎛️ Customization Options

### **Summary Length Control**
- **Slider Interface**: Easy adjustment from 10%-80% compression
- **Smart Minimums**: Ensures quality regardless of compression setting
- **Context Preservation**: Always includes important first/last sentences

### **Processing Options**
- **File Type Detection**: Automatic format recognition
- **Encoding Handling**: Supports multiple text encodings (UTF-8, Latin-1, CP1252)
- **Error Recovery**: Graceful handling of corrupted or unusual files

### **Output Formatting**
- **Section Customization**: Automatic section detection and creation
- **Professional Layout**: Clean, readable formatting for all output types
- **Timestamp Integration**: Automatic file naming with date/time stamps

## 🚀 Getting Started Guide

### **Step 1: Installation**
```bash
# Clone or download the project
cd document_summarizer

# Run the automatic setup
python start.py
```

### **Step 2: Upload Document**
- Open the web interface (automatically opens in browser)
- Drag and drop your document or click to browse
- Supported formats: PDF, DOCX, TXT

### **Step 3: Configure Settings**
- Adjust the summary length slider (10%-80%)
- Preview the extracted and cleaned text
- Review document statistics

### **Step 4: Generate Summary**
- Click "Generate Smart Summary"
- View the structured, formatted summary
- Check compression statistics and metrics

### **Step 5: Download Results**
- Choose your preferred format (TXT, PDF, DOCX)
- Download with automatic timestamped filename
- Use in your projects, studies, or work

## 💡 Tips for Best Results

### **Document Quality**
- **Clear Text**: Works best with well-formatted documents
- **Sufficient Content**: Minimum 5-10 sentences for meaningful summaries
- **Language**: Optimized for English text processing

### **Summary Settings**
- **Short Summaries (10%-20%)**: Best for quick overviews and key points
- **Balanced Summaries (20%-40%)**: Ideal for most use cases
- **Detailed Summaries (40%+)**: Comprehensive coverage with context

### **File Formats**
- **PDF**: Best for academic papers and reports
- **DOCX**: Ideal for business documents and proposals
- **TXT**: Perfect for notes, scripts, and plain text content

## 🔄 Advanced Features

### **Intelligent Content Detection**
- Automatically identifies and preserves important sentences
- Maintains logical flow and context
- Balances comprehensiveness with conciseness

### **Multi-Format Export**
- TXT files with markdown formatting for easy editing
- PDF files with professional layout for presentations
- DOCX files with proper structure for further editing

### **Real-Time Processing**
- Instant text extraction and cleaning
- Live preview of processed content
- Immediate summary generation

## 🎉 Ready to Use!

The Smart Document Summarizer is ready to transform your document processing workflow. Simply run `python start.py` and start creating intelligent, professional summaries from your documents!

---

<div align="center">
<strong>🤖 Smart Document Summarizer</strong><br>
Powered by Advanced TextRank Algorithm with Intelligent Text Cleaning<br>
<em>Transform your documents into clean, structured summaries instantly</em>
</div>
