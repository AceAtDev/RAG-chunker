"""Utility functions for the PDF to RAG converter."""

import os
import gc
import re
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging (DEBUG level)
        log_file: Optional log file path
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress verbose logs from external libraries
    if not verbose:
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('azure').setLevel(logging.WARNING)


def validate_directories(input_dir: Optional[str] = None, create_output: Optional[str] = None):
    """
    Validate input directory exists and create output directory if needed.
    
    Args:
        input_dir: Input directory to validate
        create_output: Output directory to create
        
    Raises:
        FileNotFoundError: If input directory doesn't exist
    """
    if input_dir:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    if create_output:
        output_path = Path(create_output)
        output_path.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Unicode normalization replacements
    unicode_replacements = {
        '\u012b': 'i',  # ī -> i
        '\u016b': 'u',  # ū -> u
        '\u0101': 'a',  # ā -> a
        '\u2019': "'",  # ' -> '
        '\u2018': "'",  # ' -> '
        '\u201c': '"',  # " -> "
        '\u201d': '"',  # " -> "
        '\u2013': '-',  # – -> -
        '\u2014': '-',  # — -> -
        '\u2026': '...',  # … -> ...
        '\u00a0': ' ',  # Non-breaking space -> regular space
        '\u200b': '',   # Zero-width space -> nothing
        '\u200c': '',   # Zero-width non-joiner -> nothing
        '\u200d': '',   # Zero-width joiner -> nothing
        '\u200e': '',   # Left-to-right mark -> nothing
        '\u200f': '',   # Right-to-left mark -> nothing
        '\u061c': '',   # Arabic letter mark -> nothing
        '\ufeff': '',   # Byte order mark -> nothing
    }
    
    # Apply Unicode replacements
    for unicode_char, replacement in unicode_replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
    
    # Clean up line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines -> double newline
    
    return text.strip()


def memory_cleanup():
    """Force garbage collection to free memory."""
    gc.collect()


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def format_processing_time(seconds: float) -> str:
    """Format processing time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def estimate_processing_time(file_count: int, average_file_size_mb: float) -> str:
    """Estimate processing time based on file count and size."""
    # Rough estimates based on typical processing speeds
    seconds_per_mb = 2.0  # Adjust based on your system
    estimated_seconds = file_count * average_file_size_mb * seconds_per_mb
    return format_processing_time(estimated_seconds)


def create_progress_callback(description: str):
    """Create a progress callback function for long-running operations."""
    def callback(current: int, total: int, item_name: str = ""):
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"\r{description}: {current}/{total} ({percentage:.1f}%) {item_name}", end="")
        if current >= total:
            print()  # New line when complete
    return callback


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove excessive dots and spaces
    safe_name = re.sub(r'\.+', '.', safe_name)
    safe_name = re.sub(r'\s+', '_', safe_name)
    
    # Limit length
    if len(safe_name) > 200:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:200-len(ext)] + ext
    
    return safe_name.strip('._')


def validate_api_key(api_key: str, service: str = "Azure") -> bool:
    """
    Basic validation of API key format.
    
    Args:
        api_key: API key to validate
        service: Service name for error messages
        
    Returns:
        True if key appears valid
    """
    if not api_key or not api_key.strip():
        return False
    
    # Basic length and character checks
    if len(api_key) < 10:
        return False
    
    # Check for placeholder values
    placeholder_patterns = [
        'your-key', 'api-key', 'replace-me', 'insert-key', 'key-here'
    ]
    
    if any(pattern in api_key.lower() for pattern in placeholder_patterns):
        return False
    
    return True


def get_system_info() -> Dict[str, Any]:
    """Get basic system information for debugging."""
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'disk_free_gb': round(psutil.disk_usage('.').free / (1024**3), 2)
    }


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    dependencies = {}
    
    # Core dependencies
    try:
        import PyPDF2
        dependencies['PyPDF2'] = True
    except ImportError:
        dependencies['PyPDF2'] = False
    
    try:
        import openai
        dependencies['openai'] = True
    except ImportError:
        dependencies['openai'] = False
    
    try:
        import spacy
        dependencies['spacy'] = True
        
        # Check for English model
        try:
            spacy.load('en_core_web_sm')
            dependencies['spacy_en_model'] = True
        except OSError:
            dependencies['spacy_en_model'] = False
            
    except ImportError:
        dependencies['spacy'] = False
        dependencies['spacy_en_model'] = False
    
    # Optional OCR dependencies
    try:
        import fitz  # PyMuPDF
        dependencies['pymupdf'] = True
    except ImportError:
        dependencies['pymupdf'] = False
    
    try:
        from google.cloud import vision
        dependencies['google_vision'] = True
    except ImportError:
        dependencies['google_vision'] = False
    
    try:
        import cv2
        dependencies['opencv'] = True
    except ImportError:
        dependencies['opencv'] = False
    
    return dependencies


def print_system_summary():
    """Print a summary of system information and dependencies."""
    print("=== System Information ===")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    print("\n=== Dependencies Status ===")
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"{status} {dep}")
    
    # Recommendations
    missing_core = [dep for dep, available in deps.items() 
                   if not available and dep in ['PyPDF2', 'openai', 'spacy']]
    
    if missing_core:
        print(f"\n⚠️  Missing core dependencies: {', '.join(missing_core)}")
        print("Install with: pip install -r requirements.txt")
    
    if not deps.get('spacy_en_model', False):
        print("\n⚠️  spaCy English model not found")
        print("Install with: python -m spacy download en_core_web_sm")


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = None
    
    def start(self):
        """Start progress tracking."""
        import time
        self.start_time = time.time()
        self.update(0)
    
    def update(self, current: int):
        """Update progress."""
        self.current = current
        percentage = (current / self.total) * 100 if self.total > 0 else 0
        
        elapsed_str = ""
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            elapsed_str = f" [{format_processing_time(elapsed)}]"
        
        print(f"\r{self.description}: {current}/{self.total} ({percentage:.1f}%){elapsed_str}", end="")
        
        if current >= self.total:
            print()  # New line when complete
    
    def increment(self):
        """Increment progress by 1."""
        self.update(self.current + 1)