"""
Simple startup script for the document summarization system.
Handles setup and launches the web interface.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['nltk', 'sklearn', 'numpy', 'pandas', 'streamlit', 'PyPDF2', 'docx', 'reportlab']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'docx':
                from docx import Document
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def setup_nltk():
    """Download required NLTK data."""
    print("📚 Setting up NLTK data...")
    try:
        import nltk
        required_data = ['punkt', 'stopwords', 'wordnet']
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                nltk.download(data, quiet=True)
        
        print("✅ NLTK data ready!")
        return True
    except Exception as e:
        print(f"⚠️ NLTK setup issue: {e}")
        return False

def launch_app():
    """Launch the Streamlit app."""
    print("🚀 Launching the web interface...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Thanks for using the Document Summarization System!")
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        print("\nTry running manually: streamlit run app.py")

def main():
    """Main startup function."""
    print("🌟 DOCUMENT SUMMARIZATION SYSTEM")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run this script from the project directory.")
        return
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"📋 Missing packages: {', '.join(missing)}")
        if input("Install them now? (y/n): ").lower().startswith('y'):
            if not install_dependencies():
                print("❌ Setup failed. Please install manually:")
                print("pip install -r requirements.txt")
                return
        else:
            print("❌ Cannot proceed without required packages.")
            return
    else:
        print("✅ All dependencies are installed!")
    
    # Setup NLTK
    setup_nltk()
    
    print("\n🎉 Setup complete! Launching the web interface...")
    print("📝 The app will open in your browser automatically.")
    print("🔗 If it doesn't, go to: http://localhost:8501")
    print("\n💡 Tips:")
    print("- Try the sample texts first")
    print("- Use 'Quick (TextRank)' for instant results")
    print("- Paste your own text for custom summarization")
    print("\nPress Ctrl+C to stop the server when done.\n")
    
    # Launch the app
    launch_app()

if __name__ == "__main__":
    main()