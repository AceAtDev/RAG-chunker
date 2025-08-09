# PDF to RAG Converter

A clean, modular tool for converting PDFs to text and creating RAG-ready chunks with embeddings. Perfect for building knowledge bases from document collections.


![preview gif](./public/preview.gif)



## 🚀 Features

-   **PDF to Text Conversion**: Support for both standard and OCR-based extraction
-   **Text Chunking**: Intelligent chunking with configurable size and overlap, supporting both dense (embeddings) and sparse (BM25) vector generation.
-   **Vector Generation**: Automatic embedding creation using Azure OpenAI and BM25 sparse vector generation.
-   **Multi-language Support**: Built-in translation capabilities (Arabic to English)
-   **Batch Processing**: Process entire directories with progress tracking
-   **Metadata Extraction**: Rich metadata for better search and retrieval
-   **Clean Architecture**: Modular design for easy customization and extension
## 📦 Installation

### Quick Install
```bash
git clone https://github.com/aAceAtDev/RAG-chunker.git
cd RARchunker/pdf2text2chunks
pip install -r requirements.txt
```

### Development Install
```bash
pip install -e .
```

### Optional Dependencies
```bash
# For Azure services
pip install -e ".[azure]"

# For Google Cloud services  
pip install -e ".[google]"

# For OCR capabilities
pip install -e ".[ocr]"

# For development
pip install -e ".[dev]"
```

## 🛠️ Quick Start

### 1. Setup Configuration

Create a configuration file or set environment variables:

```bash
# Create sample config
python -c "from src.config import Config; Config().create_sample_config()"
```

Edit `config.sample.json` with your credentials:

```json
{
  "azure": {
    "api_key": "your-azure-openai-api-key",
    "endpoint": "https://your-resource.openai.azure.com/",
    "api_version": "2024-05-01-preview",
    "embedding_model": "text-embedding-3-small"
  },
  "google": {
    "credentials_path": "path/to/your/google-credentials.json",
    "project_id": "your-google-project-id",
    "bucket_name": "your-bucket-name",
    "location": "us-central1"
  },
  "vectors": {
    "enable_embeddings": true,
    "embedding_retries": 3,
    "embedding_delay": 0.06,
    "enable_bm25": true,
    "bm25_max_dim": 1000,
    "bm25_k1": 1.2,
    "bm25_b": 0.75
  },
  "processing": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "max_workers": 4,
    "default_language": "en",
    "supported_languages": [
      "en",
      "ar"
    ]
  }
}
```

### 2. Convert PDFs to Text

```bash
# Convert all PDFs in a directory
python main.py convert --input pdfs/ --output text/
```

### 3. Create RAG Chunks

```bash
# Create chunks with both embeddings and BM25 (default)
python main.py chunk --input text/ --output chunks/ --azure-key YOUR_API_KEY

# Create chunks with embeddings only
python main.py chunk --input text/ --output chunks/ --azure-key YOUR_API_KEY --embeddings-only

# Create chunks with BM25 only
python main.py chunk --input text/ --output chunks/ --bm25-only

# Custom chunk settings
python main.py chunk --input text/ --output chunks/ \
  --azure-key YOUR_API_KEY \
  --chunk-size 1024 \
  --chunk-overlap 100
```

### 4. Translate Text Files

```bash

# Translate text files from Arabic to English using Azure
python main.py translate --input text_ar/ --output text_en/ --translator azure --key YOUR_API_KEY

```

### 5. Full Pipeline

```bash
# Run complete pipeline: PDF → Text → Chunks
python main.py pipeline --input pdfs/ --azure-key YOUR_API_KEY

# With translation
python main.py pipeline --input pdfs/ --azure-key YOUR_API_KEY --translate
```

## 📖 Usage Examples

### Basic PDF Processing

```python
from src.pdf_converter import PDFConverter

converter = PDFConverter(input_dir="pdfs", output_dir="text")
results = converter.process_directory()

print(f"Converted {results['successful']} files")
```

### Creating Chunks with Embeddings

```python
from src.text_chunker import TextChunker

chunker = TextChunker(
    input_dir="text",
    output_dir="chunks",
    azure_api_key="your-key",
    chunk_size=512,
    chunk_overlap=50,
    enable_embeddings=True, 
    enable_bm25=True,       
    bm25_max_dim=1000       
)

results = chunker.process_directory()
print(f"Created {results['total_chunks']} chunks")
```

### Translation

```python
from src.translator import Translator

translator = Translator(
    service="azure",
    api_key="your-key",
    source_lang="ar",
    target_lang="en",
    input_dir="text_ar",
    output_dir="text_en"
)

results = translator.process_directory()
```

## 🔧 Configuration

### Environment Variables

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export CHUNK_SIZE=512
export CHUNK_OVERLAP=50
export MAX_WORKERS=4
export ENABLE_EMBEDDINGS=true
export ENABLE_BM25=true
export BM25_MAX_DIM=1000
```

### Configuration File

The tool looks for `config.json` in the current directory. You can also specify a custom config file:

```python
from src.config import Config
config = Config("my-config.json")
```

## 📄 Output Format

### Text Files
Standard UTF-8 text files with cleaned and normalized content.

### Chunk Files
JSON files containing structured chunk data:

```json
{
  "id": "document_chunk_0001",
  "content": "chunk text content...",
  "embedding": [0.1, 0.2, ...],
  "metadata": {
    "file": {
      "filename": "document.pdf",
      "title": "Document Title",
      "author": "Author Name"
    },
    "chunk": {
      "chunk_index": 0,
      "total_chunks": 25,
      "word_count": 487,
      "character_count": 2451
    },
    "content": {
      "categories": ["..."],
      "entities": ["..."],
      "key_phrases": ["..."]
    }
  }
}
```

## 🔍 Advanced Features

### Custom Chunking Strategy

```python
class CustomChunker(TextChunker):
    def create_chunks(self, text: str, filename: str) -> List[Dict[str, Any]]:
        # Your custom chunking logic
        # Remember to call self._get_embeddings and self._generate_bm25_vectors
        # and include them in the returned chunk data if enabled.
        custom_chunks_data = []
        # ... your logic ...
        return custom_chunks_data
chunker = CustomChunker(...)
```

### Metadata Enhancement

```python
def custom_metadata_extractor(text: str) -> Dict:
    # Extract domain-specific metadata
    return metadata

chunker.extract_metadata_from_text = custom_metadata_extractor
```

### OCR for Scanned PDFs

```python
from src.ocr_converter import OCRConverter

ocr_converter = OCRConverter(
    output_dir="text",
    bucket_name="your-gcs-bucket",  # For Google Vision API
    credentials_path="path/to/credentials.json"
)

results = ocr_converter.process_bucket()
```

## 📊 Performance

### Typical Processing Speeds
- **Standard PDFs**: ~10-50 pages/minute
- **OCR PDFs**: ~5-15 pages/minute  
- **Chunking**: ~1000 chunks/minute
- **Embedding**: ~500 chunks/minute (depends on API limits)

### Memory Usage
- **Standard processing**: ~100-500MB
- **Large PDFs**: Up to 2GB (automatic cleanup implemented)
- **Batch processing**: Scales with worker count

## 🐛 Troubleshooting

### Common Issues

**1. spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

**2. Azure API errors**
- Check your API key and endpoint
- Verify rate limits
- Ensure model deployment name is correct

**3. Memory issues with large PDFs**
- Reduce `max_workers`
- Process files individually
- Increase system memory

**4. OCR dependencies**
```bash
# For Tesseract
sudo apt-get install tesseract-ocr

# For OpenCV
pip install opencv-python
```

### Debug Mode

```bash
python main.py convert --input pdfs/ --output text/ --verbose
```

### System Check

```python
from src.utils import print_system_summary
print_system_summary()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/yourusername/pdf-to-rag-converter.git
cd pdf-to-rag-converter
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Type checking
mypy src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyPDF2, OpenAI, spaCy, and other fantastic open-source libraries
- Inspired by the need for better document processing in RAG systems
- Special thanks to the Arabic NLP community for guidance on multilingual processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/pdf-to-rag-converter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pdf-to-rag-converter/discussions)
- **Email**: aceatdeveloping@gmail.com

---

**Made with ❤️ for the RAG community**
