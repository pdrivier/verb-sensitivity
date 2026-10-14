

import os
import shutil
from pathlib import Path

def clear_huggingface_cache():
    """Clear Hugging Face model cache from the default cache directory."""
    # Get the cache directory path
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    
    # Check if the cache directory exists
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    try:
        # List all items in the cache directory
        models = os.listdir(cache_dir)
        
        # Remove directories that contain "model" in their name
        for model_dir in models:
            if "model" in model_dir.lower():
                model_path = cache_dir / model_dir
                if model_path.is_dir():
                    print(f"Removing: {model_path}")
                    shutil.rmtree(model_path)
                    
        print("Cache clearing completed successfully!")
        
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"Error clearing cache: {e}")

# Call the function
if __name__ == "__main__":
    clear_huggingface_cache()