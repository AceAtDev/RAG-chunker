"""Text chunking with embeddings and BM25 sparse vectors for RAG systems."""

import os
import json
import logging
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib

import spacy
import numpy as np
from openai import AzureOpenAI
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from .config import Config
from .utils import clean_text, memory_cleanup

logger = logging.getLogger(__name__)


class TextChunker:
    """Text chunker with embedding generation and BM25 sparse vectors for RAG systems."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        api_version: str = "2024-05-01-preview",
        embedding_model: str = "text-embedding-3-small",
        enable_embeddings: bool = True,
        enable_bm25: bool = True,
        bm25_max_dim: int = 1000,
        embedding_retries: int = 3,
        embedding_delay: float = 0.06
    ):
        """
        Initialize the text chunker.

        Args:
            input_dir: Directory containing text files
            output_dir: Directory where chunk files will be saved
            azure_api_key: Azure OpenAI API key (required if enable_embeddings=True)
            azure_endpoint: Azure OpenAI endpoint (optional)
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            api_version: Azure OpenAI API version
            embedding_model: Name of the embedding model
            enable_embeddings: Whether to generate vector embeddings
            enable_bm25: Whether to compute BM25 sparse vectors
            bm25_max_dim: Maximum dimensions for BM25 sparse vectors
            embedding_retries: Number of retry attempts for embedding generation
            embedding_delay: Delay between embedding requests (rate limiting)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_embeddings = enable_embeddings
        self.enable_bm25 = enable_bm25
        self.bm25_max_dim = bm25_max_dim
        self.embedding_retries = embedding_retries
        self.embedding_delay = embedding_delay

        # Validate configuration
        if not self.enable_embeddings and not self.enable_bm25:
            raise ValueError("At least one of enable_embeddings or enable_bm25 must be True")
        
        if self.enable_embeddings and not azure_api_key:
            raise ValueError("Azure OpenAI API key is required when enable_embeddings=True")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Initialize spaCy for NLP processing
        self._init_spacy()

        # Initialize Azure OpenAI client (only if embeddings are enabled)
        self.client = None
        if self.enable_embeddings:
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint or self._get_default_endpoint()
            )
            self.embedding_model = embedding_model

            # Test Azure OpenAI connection immediately after initialization
            logger.info("Testing Azure OpenAI configuration...")
            if not self._test_azure_connection():
                raise ConnectionError("Azure OpenAI connection test failed. Please check your configuration.")
            logger.info("✅ Azure OpenAI connection test passed!")
        else:
            logger.info("Embeddings disabled - skipping Azure OpenAI initialization")
            self.embedding_model = None

        # Rate limiting (only used if embeddings are enabled)
        self.request_delay = embedding_delay

    def _test_azure_connection(self) -> bool:
        """
        Test Azure OpenAI connection and configuration.
        Only called if embeddings are enabled.

        Returns:
            True if connection test passes, False otherwise
        """
        if not self.enable_embeddings or not self.client:
            return True  # Skip test if embeddings are disabled
        
        # ANSI color codes for logging in red
        RED = '\033[91m'
        RESET = '\033[0m'
        
        try:
            logger.info("🔍 Running Azure OpenAI connection test...")

            # Test 1: Basic client validation
            if not self.client:
                logger.error("❌ Azure OpenAI client is not initialized")
                return False

            # Test 2: Simple embedding generation
            test_text = "Connection test for Azure OpenAI embedding service."
            logger.info(f"📝 Testing embedding generation with text: '{test_text[:30]}...'")

            start_time = time.time()

            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=test_text
                )

                end_time = time.time()
                response_time = end_time - start_time

                # Validate response
                if not response or not response.data:
                    logger.error("❌ Empty response from embedding API")
                    return False

                embedding = response.data[0].embedding

                if not embedding or len(embedding) == 0:
                    logger.error("❌ No embedding data returned")
                    return False

                # Log success details
                logger.info(f"✅ Embedding test successful!")
                logger.info(f"   📊 Response time: {response_time:.2f} seconds")
                logger.info(f"   📐 Embedding dimensions: {len(embedding)}")
                logger.info(f"   🔢 First few values: {embedding[:3]}")
                logger.info(f"   🤖 Model used: {response.model}")
                logger.info(f"   💰 Tokens used: {response.usage.total_tokens}")

                # Test 3: Validate embedding values are reasonable
                if not all(isinstance(val, (int, float)) for val in embedding[:5]):
                    logger.error("❌ Embedding contains invalid values")
                    return False

                # Test 4: Check embedding dimension is expected
                expected_dimensions = {
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072,
                    "text-embedding-ada-002": 1536
                }

                expected_dim = expected_dimensions.get(self.embedding_model)
                if expected_dim and len(embedding) != expected_dim:
                    logger.warning(f"⚠️  Unexpected embedding dimension: got {len(embedding)}, expected {expected_dim}")

                logger.info("✅ All connection tests passed!")
                return True

            except Exception as api_error:
                logger.error(f"❌ Embedding API call failed: {api_error}")

                # Provide specific error guidance
                error_str = str(api_error).lower()

                if "authentication" in error_str or "unauthorized" in error_str:
                    logger.error("   🔑 Issue: Invalid API key")
                    logger.error("   💡 Solution: Check your Azure OpenAI API key")

                elif "not found" in error_str or "404" in error_str:
                    logger.error("   🎯 Issue: Model or endpoint not found")
                    logger.error(f"   💡 Solution: Verify model '{self.embedding_model}' is deployed")

                elif "connection" in error_str or "timeout" in error_str:
                    logger.error("   🌐 Issue: Network connectivity problem")
                    logger.error("   💡 Solution: Check internet connection and endpoint URL")

                elif "rate" in error_str or "quota" in error_str:
                    logger.error("   ⏱️  Issue: Rate limiting or quota exceeded")
                    logger.error("   💡 Solution: Wait and retry, or check your quota")

                else:
                    logger.error(f"   ❓ Unknown error type: {api_error}")
                
                # Log the raw error in red, as requested
                logger.error(f"{RED}RAW ERROR: {repr(api_error)}{RESET}")
                return False

        except Exception as e:
            logger.error(f"❌ Connection test failed with unexpected error: {e}")
            # Log the raw error in red, as requested
            logger.error(f"{RED}RAW ERROR: {repr(e)}{RESET}")
            return False

    def _init_spacy(self):
        """Initialize spaCy model."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.info("Downloading spaCy English model...")
            try:
                spacy.cli.download('en_core_web_sm')
                self.nlp = spacy.load('en_core_web_sm')
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                logger.warning("Using basic sentence splitting instead of spaCy")
                self.nlp = None

    def _get_default_endpoint(self) -> str:
        """Get default Azure endpoint from environment or config."""
        config = Config()
        return config.azure.endpoint or "https://your-resource.openai.azure.com/"

    def generate_chunk_id(self, filename: str, chunk_number: int) -> str:
        """Generate unique ID for a chunk."""
        clean_filename = Path(filename).stem
        return f"{clean_filename}_chunk_{chunk_number:04d}"

    def get_embedding(self, text: str, retries: int = None) -> List[float]:
        """
        Generate embedding for the given text.
        Only called if embeddings are enabled.

        Args:
            text: Text to embed
            retries: Number of retry attempts (uses instance default if None)

        Returns:
            List of embedding values, or empty list if embeddings disabled
        """
        if not self.enable_embeddings or not self.client:
            return []
        
        retries = retries or self.embedding_retries

        for attempt in range(retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                time.sleep(self.request_delay)  # Rate limiting
                return response.data[0].embedding

            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Embedding generation failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embedding after {retries} attempts: {e}")
                    return []

    def preprocess_text_for_bm25(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 tokenization.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of tokens
        """
        return text.lower().split()

    def compute_bm25_sparse_vector(self, bm25: BM25Okapi, query_tokens: List[str]) -> Dict[str, List]:
        """
        Compute a sparse BM25 vector representation.

        Args:
            bm25: Fitted BM25Okapi instance
            query_tokens: Tokenized query text

        Returns:
            Dict with 'indices' and 'values' for a sparse vector
        """
        # Get BM25 scores for the query tokens
        scores = bm25.get_scores(query_tokens)

        # Sort scores in descending order and get top indices
        sorted_indices = np.argsort(scores)[::-1]

        # Take top scores, limiting to max_dim
        top_indices = sorted_indices[:self.bm25_max_dim]
        top_scores = scores[top_indices]

        # Filter out zero scores
        non_zero_mask = top_scores > 0
        indices = top_indices[non_zero_mask]
        values = top_scores[non_zero_mask]

        return {
            "indices": indices.tolist(),
            "values": values.tolist()
        }

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy or regex fallback."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback to regex-based sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> List[str]:
        """
        Create overlapping chunks from text.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence would exceed chunk size and we have content
            if current_size + sentence_words > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))

                # Create overlap for next chunk
                overlap_chunk = []
                overlap_size = 0

                # Add sentences from the end for overlap
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_size + sent_words <= self.chunk_overlap:
                        overlap_chunk.insert(0, sent)
                        overlap_size += sent_words
                    else:
                        break

                current_chunk = overlap_chunk
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_words

        # Add final chunk if there's content
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text content."""
        metadata = {
            'word_count': len(text.split()),
            'character_count': len(text),
            'sentence_count': len(self.split_into_sentences(text)),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'language': 'en',  # Default, could be detected
            'content_type': 'text'
        }

        # Extract potential tags using spaCy if available
        if self.nlp:
            doc = self.nlp(text[:1000])  # Analyze first 1000 chars for performance

            # Named entities
            entities = list(set([ent.text.lower() for ent in doc.ents if len(ent.text) > 2]))

            # Key noun phrases
            noun_phrases = list(set([
                chunk.text.lower() for chunk in doc.noun_chunks 
                if len(chunk.text.split()) <= 3 and len(chunk.text) > 3
            ]))

            metadata['entities'] = entities[:10]  # Limit to top 10
            metadata['key_phrases'] = noun_phrases[:10]

        return metadata

    def extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file."""
        stat = file_path.stat()

        # Try to parse filename for structured info
        filename_parts = file_path.stem.replace('_', ' ').replace('-', ' ')

        return {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': stat.st_size,
            'created_date': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'processed_date': datetime.now().isoformat(),
            'title': filename_parts,  # Use filename as title
            'source_type': 'text_file'
        }

    def determine_content_categories(self, text: str) -> List[str]:
        """Determine content categories based on text analysis."""
        text_lower = text.lower()
        categories = []

        # Islamic/Religious content detection
        islamic_terms = {
            'religious': ['allah', 'prophet', 'islam', 'muslim', 'quran', 'hadith', 'prayer', 'mosque'],
            'jurisprudence': ['halal', 'haram', 'ruling', 'permissible', 'forbidden', 'fiqh'],
            'theology': ['belief', 'faith', 'creed', 'aqeedah', 'tawheed'],
            'history': ['historical', 'battle', 'expedition', 'companion', 'sahaba'],
            'ethics': ['character', 'manners', 'akhlaq', 'morality', 'conduct']
        }

        for category, terms in islamic_terms.items():
            if any(term in text_lower for term in terms):
                categories.append(category)

        # Academic content detection
        academic_indicators = ['research', 'study', 'analysis', 'conclusion', 'methodology']
        if any(indicator in text_lower for indicator in academic_indicators):
            categories.append('academic')

        # Narrative content detection
        narrative_indicators = ['story', 'narrative', 'tale', 'account', 'narrated']
        if any(indicator in text_lower for indicator in narrative_indicators):
            categories.append('narrative')

        return categories if categories else ['general']

    def create_chunk_document(
        self, 
        chunk_text: str, 
        chunk_id: str, 
        chunk_index: int,
        total_chunks: int,
        file_metadata: Dict[str, Any],
        content_metadata: Dict[str, Any],
        bm25_vector: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """Create a complete chunk document with all metadata."""

        # Generate embedding (only if enabled)
        embedding = self.get_embedding(chunk_text) if self.enable_embeddings else None

        # Create chunk-specific metadata
        chunk_metadata = {
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_position': f"{chunk_index + 1}/{total_chunks}",
            'is_first_chunk': chunk_index == 0,
            'is_last_chunk': chunk_index == total_chunks - 1,
            'word_count': len(chunk_text.split()),
            'character_count': len(chunk_text),
            'content_hash': hashlib.md5(chunk_text.encode()).hexdigest()
        }

        # Create document structure
        document = {
            'id': chunk_id,
            'content': chunk_text,
            'metadata': {
                'file': file_metadata,
                'chunk': chunk_metadata,
                'content': content_metadata,
                'categories': self.determine_content_categories(chunk_text),
                'processing': {
                    'chunker_version': '1.0.0',
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'embeddings_enabled': self.enable_embeddings,
                    'embedding_model': self.embedding_model,
                    'bm25_enabled': self.enable_bm25,
                    'bm25_max_dim': self.bm25_max_dim if self.enable_bm25 else None,
                    'processed_at': datetime.now().isoformat()
                }
            }
        }

        # Add embedding if enabled
        if self.enable_embeddings and embedding:
            document['embedding'] = embedding

        # Add BM25 vector if available
        if bm25_vector is not None:
            document['bm25_vector'] = bm25_vector

        return document

    def process_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single text file into chunks."""
        result = {
            'filename': file_path.name,
            'status': 'failed',
            'error': None,
            'output_path': None,
            'chunk_count': 0
        }

        try:
            logger.debug(f"Processing text file: {file_path}")

            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                result['error'] = "Empty text file"
                return result

            # Clean text
            text = clean_text(text)

            # Extract file and content metadata
            file_metadata = self.extract_file_metadata(file_path)
            content_metadata = self.extract_metadata_from_text(text)

            # Create chunks
            chunks = self.create_chunks(text)

            if not chunks:
                result['error'] = "No chunks created from text"
                return result

            # Prepare BM25 if enabled
            bm25_index = None
            chunk_tokens = None
            
            if self.enable_bm25:
                logger.info(f"Computing BM25 index for {file_path.name}")
                chunk_tokens = [self.preprocess_text_for_bm25(chunk) for chunk in chunks]
                bm25_index = BM25Okapi(chunk_tokens)

            # Create chunk documents
            chunk_documents = []

            # Determine progress description based on enabled features
            features = []
            if self.enable_embeddings:
                features.append("embeddings")
            if self.enable_bm25:
                features.append("BM25")
            desc = f"Creating chunks ({', '.join(features)}) for {file_path.name}"

            with tqdm(
                total=len(chunks), 
                desc=desc,
                unit="chunk",
                leave=False
            ) as pbar:
                for i, chunk_text in enumerate(chunks):
                    chunk_id = self.generate_chunk_id(file_path.name, i)

                    # Compute BM25 vector if enabled
                    bm25_vector = None
                    if self.enable_bm25 and bm25_index is not None:
                        bm25_vector = self.compute_bm25_sparse_vector(bm25_index, chunk_tokens[i])

                    chunk_doc = self.create_chunk_document(
                        chunk_text=chunk_text,
                        chunk_id=chunk_id,
                        chunk_index=i,
                        total_chunks=len(chunks),
                        file_metadata=file_metadata,
                        content_metadata=content_metadata,
                        bm25_vector=bm25_vector
                    )

                    chunk_documents.append(chunk_doc)
                    pbar.update(1)

            # Save chunks to JSON file
            output_path = self.output_dir / f"{file_path.stem}_chunks.json"

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_documents, f, indent=2, ensure_ascii=False)

            result.update({
                'status': 'success',
                'output_path': str(output_path),
                'chunk_count': len(chunks)
            })

            # Generate status message
            features = []
            if self.enable_embeddings:
                features.append("embeddings")
            if self.enable_bm25:
                features.append("BM25 vectors")
            feature_status = f"with {' and '.join(features)}" if features else "without vectors"
            
            logger.info(f"Successfully processed {file_path.name}: {len(chunks)} chunks created {feature_status}")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing {file_path.name}: {e}")

        return result

    def get_text_files(self) -> List[Path]:
        """Get list of text files in the input directory."""
        text_files = []
        for ext in ['*.txt', '*.TXT']:
            text_files.extend(self.input_dir.glob(ext))
        return sorted(text_files)

    def process_directory(self) -> Dict[str, Any]:
        """Process all text files in the input directory."""
        text_files = self.get_text_files()

        if not text_files:
            logger.warning(f"No text files found in {self.input_dir}")
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'processed_files': [],
                'processing_time': 0,
                'total_chunks': 0
            }

        # Generate status message for logging
        features = []
        if self.enable_embeddings:
            features.append("embeddings")
        if self.enable_bm25:
            features.append("BM25 vectors")
        feature_status = f"with {' and '.join(features)}" if features else "without vectors"
        
        logger.info(f"Found {len(text_files)} text files to process {feature_status}")

        results = {
            'total_files': len(text_files),
            'successful': 0,
            'failed': 0,
            'processed_files': [],
            'processing_time': 0,
            'total_chunks': 0,
            'embeddings_enabled': self.enable_embeddings,
            'bm25_enabled': self.enable_bm25
        }

        start_time = time.time()

        # Process files with progress bar
        with tqdm(total=len(text_files), desc="Processing text files", unit="file") as pbar:
            for file_path in text_files:
                result = self.process_text_file(file_path)
                results['processed_files'].append(result)

                if result['status'] == 'success':
                    results['successful'] += 1
                    results['total_chunks'] += result['chunk_count']
                else:
                    results['failed'] += 1

                pbar.update(1)
                pbar.set_postfix(
                    success=results['successful'],
                    failed=results['failed'],
                    chunks=results['total_chunks']
                )

                # Memory cleanup
                memory_cleanup()

        results['processing_time'] = time.time() - start_time

        # Log summary
        logger.info(f"Processing completed in {results['processing_time']:.2f} seconds")
        logger.info(f"Successfully processed: {results['successful']} files")
        logger.info(f"Failed processing: {results['failed']} files")
        logger.info(f"Total chunks created: {results['total_chunks']}")
        
        # Log feature-specific status
        if self.enable_embeddings:
            logger.info("✅ Vector embeddings generated for all chunks")
        if self.enable_bm25:
            logger.info("✅ BM25 sparse vectors generated for all chunks")

        return results