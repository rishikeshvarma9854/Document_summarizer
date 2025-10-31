"""
Script to clear Hugging Face model cache and free up disk space.
"""

import os
import shutil
from pathlib import Path

def get_cache_size(path):
    """Get the size of a directory in MB."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total_size / (1024 * 1024)  # Convert to MB

def clear_huggingface_cache():
    """Clear Hugging Face model cache."""
    
    # Common cache locations
    cache_paths = [
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch" / "hub",
        Path(os.environ.get('TRANSFORMERS_CACHE', Path.home() / ".cache" / "huggingface")),
    ]
    
    total_freed = 0
    
    print("🔍 Checking Hugging Face cache locations...")
    
    for cache_path in cache_paths:
        if cache_path.exists():
            size_mb = get_cache_size(cache_path)
            print(f"📁 Found cache: {cache_path}")
            print(f"💾 Size: {size_mb:.1f} MB")
            
            if size_mb > 0:
                response = input(f"❓ Delete {cache_path}? (y/n): ").lower().strip()
                if response == 'y':
                    try:
                        shutil.rmtree(cache_path)
                        print(f"✅ Deleted {cache_path} - Freed {size_mb:.1f} MB")
                        total_freed += size_mb
                    except Exception as e:
                        print(f"❌ Error deleting {cache_path}: {e}")
                else:
                    print(f"⏭️ Skipped {cache_path}")
            print()
    
    # Also check for PyTorch cache
    torch_cache = Path.home() / ".cache" / "torch"
    if torch_cache.exists():
        size_mb = get_cache_size(torch_cache)
        if size_mb > 0:
            print(f"📁 Found PyTorch cache: {torch_cache}")
            print(f"💾 Size: {size_mb:.1f} MB")
            response = input(f"❓ Delete PyTorch cache? (y/n): ").lower().strip()
            if response == 'y':
                try:
                    shutil.rmtree(torch_cache)
                    print(f"✅ Deleted PyTorch cache - Freed {size_mb:.1f} MB")
                    total_freed += size_mb
                except Exception as e:
                    print(f"❌ Error deleting PyTorch cache: {e}")
    
    print(f"\n🎉 Total space freed: {total_freed:.1f} MB ({total_freed/1024:.2f} GB)")
    
    if total_freed == 0:
        print("ℹ️ No cache found or nothing to delete.")

def show_cache_info():
    """Show information about current cache usage."""
    cache_paths = [
        ("Hugging Face Hub", Path.home() / ".cache" / "huggingface" / "hub"),
        ("Hugging Face Datasets", Path.home() / ".cache" / "huggingface" / "datasets"),
        ("PyTorch Hub", Path.home() / ".cache" / "torch" / "hub"),
        ("Transformers", Path.home() / ".cache" / "huggingface" / "transformers"),
    ]
    
    print("📊 Current Cache Usage:")
    print("-" * 50)
    
    total_size = 0
    for name, path in cache_paths:
        if path.exists():
            size_mb = get_cache_size(path)
            total_size += size_mb
            print(f"{name:20}: {size_mb:8.1f} MB")
        else:
            print(f"{name:20}: {'Not found':>8}")
    
    print("-" * 50)
    print(f"{'Total':20}: {total_size:8.1f} MB ({total_size/1024:.2f} GB)")

if __name__ == "__main__":
    print("🧹 Hugging Face Model Cache Cleaner")
    print("=" * 40)
    
    show_cache_info()
    print()
    
    response = input("❓ Do you want to clear the cache? (y/n): ").lower().strip()
    if response == 'y':
        clear_huggingface_cache()
    else:
        print("👍 Cache preserved.")
        
    print("\n💡 Tip: You can also manually delete the folders:")
    print(f"   📁 {Path.home() / '.cache' / 'huggingface'}")
    print(f"   📁 {Path.home() / '.cache' / 'torch'}")