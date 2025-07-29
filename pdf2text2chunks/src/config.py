"""Configuration management for PDF to RAG converter."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class AzureConfig:
    """Azure service configuration."""
    api_key: str = ""
    endpoint: str = ""
    api_version: str = "2024-05-01-preview"
    embedding_model: str = "text-embedding-3-small"
    translator_location: str = "eastus"


@dataclass
class GoogleConfig:
    """Google Cloud configuration."""
    credentials_path: str = ""
    project_id: str = ""
    bucket_name: str = ""
    location: str = "us-central1"


@dataclass
class VectorConfig:
    """Vector generation configuration."""
    # Embedding settings
    enable_embeddings: bool = True
    embedding_retries: int = 3
    embedding_delay: float = 0.06  # Rate limiting delay in seconds
    
    # BM25 settings
    enable_bm25: bool = True
    bm25_max_dim: int = 1000
    bm25_k1: float = 1.2  # BM25 parameter k1
    bm25_b: float = 0.75  # BM25 parameter b
    
    def __post_init__(self):
        """Validate vector configuration."""
        if not self.enable_embeddings and not self.enable_bm25:
            raise ValueError("At least one of enable_embeddings or enable_bm25 must be True")


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_workers: int = 4
    default_language: str = "en"
    supported_languages: list = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en", "ar"]


class Config:
    """Main configuration class."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.azure = AzureConfig()
        self.google = GoogleConfig()
        self.vectors = VectorConfig()
        self.processing = ProcessingConfig()
        
        # Load configuration from file or environment
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        # Try to load from config file first
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self._update_from_dict(config_data)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
        
        # Override with environment variables
        self._load_from_env()
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        if "azure" in config_data:
            azure_data = config_data["azure"]
            for key, value in azure_data.items():
                if hasattr(self.azure, key):
                    setattr(self.azure, key, value)
        
        if "google" in config_data:
            google_data = config_data["google"]
            for key, value in google_data.items():
                if hasattr(self.google, key):
                    setattr(self.google, key, value)
        
        if "vectors" in config_data:
            vectors_data = config_data["vectors"]
            for key, value in vectors_data.items():
                if hasattr(self.vectors, key):
                    setattr(self.vectors, key, value)
        
        if "processing" in config_data:
            processing_data = config_data["processing"]
            for key, value in processing_data.items():
                if hasattr(self.processing, key):
                    setattr(self.processing, key, value)
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Azure configuration
        self.azure.api_key = os.getenv("AZURE_OPENAI_API_KEY", self.azure.api_key)
        self.azure.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", self.azure.endpoint)
        self.azure.api_version = os.getenv("AZURE_OPENAI_API_VERSION", self.azure.api_version)
        self.azure.embedding_model = os.getenv("AZURE_EMBEDDING_MODEL", self.azure.embedding_model)
        self.azure.translator_location = os.getenv("AZURE_TRANSLATOR_LOCATION", self.azure.translator_location)
        
        # Google configuration
        self.google.credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", self.google.credentials_path)
        self.google.project_id = os.getenv("GOOGLE_PROJECT_ID", self.google.project_id)
        self.google.bucket_name = os.getenv("GOOGLE_BUCKET_NAME", self.google.bucket_name)
        self.google.location = os.getenv("GOOGLE_LOCATION", self.google.location)
        
        # Vector configuration
        if os.getenv("ENABLE_EMBEDDINGS"):
            self.vectors.enable_embeddings = os.getenv("ENABLE_EMBEDDINGS").lower() == "true"
        if os.getenv("ENABLE_BM25"):
            self.vectors.enable_bm25 = os.getenv("ENABLE_BM25").lower() == "true"
        if os.getenv("BM25_MAX_DIM"):
            self.vectors.bm25_max_dim = int(os.getenv("BM25_MAX_DIM"))
        if os.getenv("EMBEDDING_RETRIES"):
            self.vectors.embedding_retries = int(os.getenv("EMBEDDING_RETRIES"))
        
        # Processing configuration
        if os.getenv("CHUNK_SIZE"):
            self.processing.chunk_size = int(os.getenv("CHUNK_SIZE"))
        if os.getenv("CHUNK_OVERLAP"):
            self.processing.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        if os.getenv("MAX_WORKERS"):
            self.processing.max_workers = int(os.getenv("MAX_WORKERS"))
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        file_path = file_path or self.config_file
        config_dict = {
            "azure": asdict(self.azure),
            "google": asdict(self.google),
            "vectors": asdict(self.vectors),
            "processing": asdict(self.processing)
        }
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def create_sample_config(self, file_path: str = "config.sample.json"):
        """Create a sample configuration file."""
        sample_config = {
            "azure": {
                "api_key": "your-azure-openai-api-key",
                "endpoint": "https://your-resource.openai.azure.com/",
                "api_version": "2024-05-01-preview",
                "embedding_model": "text-embedding-3-small",
                "translator_location": "eastus"
            },
            "google": {
                "credentials_path": "path/to/your/google-credentials.json",
                "project_id": "your-google-project-id",
                "bucket_name": "your-bucket-name",
                "location": "us-central1"
            },
            "vectors": {
                "enable_embeddings": True,
                "embedding_retries": 3,
                "embedding_delay": 0.06,
                "enable_bm25": True,
                "bm25_max_dim": 1000,
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            },
            "processing": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "max_workers": 4,
                "default_language": "en",
                "supported_languages": ["en", "ar"]
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"Sample configuration file created: {file_path}")
        print("Please edit this file with your actual credentials and settings.")
        print("\nVector Configuration Options:")
        print("- enable_embeddings: true/false - Generate vector embeddings")
        print("- enable_bm25: true/false - Generate BM25 sparse vectors")
        print("- Both can be enabled simultaneously (default)")
        print("- At least one must be enabled")
    
    def validate(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate vector configuration first
        if not self.vectors.enable_embeddings and not self.vectors.enable_bm25:
            issues.append("At least one of enable_embeddings or enable_bm25 must be True")
        
        # Validate Azure config (only if embeddings are enabled)
        if self.vectors.enable_embeddings:
            if not self.azure.api_key:
                issues.append("Azure OpenAI API key is required when embeddings are enabled")
            if not self.azure.endpoint:
                issues.append("Azure OpenAI endpoint is required when embeddings are enabled")
        
        # Validate Google config (if needed)
        if self.google.credentials_path and not os.path.exists(self.google.credentials_path):
            issues.append(f"Google credentials file not found: {self.google.credentials_path}")
        
        # Validate processing config
        if self.processing.chunk_size <= 0:
            issues.append("Chunk size must be positive")
        if self.processing.chunk_overlap < 0:
            issues.append("Chunk overlap cannot be negative")
        if self.processing.chunk_overlap >= self.processing.chunk_size:
            issues.append("Chunk overlap must be less than chunk size")
        if self.processing.max_workers <= 0:
            issues.append("Max workers must be positive")
        
        # Validate vector-specific settings
        if self.vectors.enable_bm25:
            if self.vectors.bm25_max_dim <= 0:
                issues.append("BM25 max dimensions must be positive")
            if not (0 < self.vectors.bm25_k1 <= 3):
                issues.append("BM25 k1 parameter should be between 0 and 3")
            if not (0 <= self.vectors.bm25_b <= 1):
                issues.append("BM25 b parameter should be between 0 and 1")
        
        if self.vectors.enable_embeddings:
            if self.vectors.embedding_retries <= 0:
                issues.append("Embedding retries must be positive")
            if self.vectors.embedding_delay < 0:
                issues.append("Embedding delay cannot be negative")
        
        return issues
    
    def get_vector_mode_description(self) -> str:
        """Get a description of the current vector generation mode."""
        if self.vectors.enable_embeddings and self.vectors.enable_bm25:
            return "Hybrid mode: Both embeddings and BM25 sparse vectors"
        elif self.vectors.enable_embeddings:
            return "Embeddings only: Dense vector embeddings"
        elif self.vectors.enable_bm25:
            return "BM25 only: Sparse BM25 vectors"
        else:
            return "Invalid: No vector generation enabled"
    
    def __str__(self):
        """String representation of configuration."""
        return f"""Configuration:
Azure:
  API Key: {'*' * len(self.azure.api_key) if self.azure.api_key else 'Not set'}
  Endpoint: {self.azure.endpoint or 'Not set'}
  Model: {self.azure.embedding_model}

Google:
  Project ID: {self.google.project_id or 'Not set'}
  Credentials: {self.google.credentials_path or 'Not set'}

Vectors:
  Mode: {self.get_vector_mode_description()}
  Embeddings: {'Enabled' if self.vectors.enable_embeddings else 'Disabled'}
  BM25: {'Enabled' if self.vectors.enable_bm25 else 'Disabled'}
  BM25 Max Dimensions: {self.vectors.bm25_max_dim}

Processing:
  Chunk Size: {self.processing.chunk_size}
  Chunk Overlap: {self.processing.chunk_overlap}
  Max Workers: {self.processing.max_workers}
"""


def get_config(config_file: Optional[str] = None) -> Config:
    """Get configuration instance."""
    return Config(config_file)


if __name__ == "__main__":
    # Create sample config for testing
    config = Config()
    config.create_sample_config()
    print(config)