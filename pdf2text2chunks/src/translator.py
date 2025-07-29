"""Translation services for text files."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from tqdm import tqdm

from .utils import clean_text, memory_cleanup

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """Abstract base class for translation services."""
    
    def __init__(self, api_key: str, source_lang: str = "ar", target_lang: str = "en"):
        self.api_key = api_key
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    @abstractmethod
    def translate_text(self, text: str) -> str:
        """Translate text from source to target language."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test the translation service connection."""
        pass


class AzureTranslator(BaseTranslator):
    """Azure Cognitive Services Translator."""
    
    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.cognitive.microsofttranslator.com/",
        location: str = "eastus",
        source_lang: str = "ar",
        target_lang: str = "en"
    ):
        super().__init__(api_key, source_lang, target_lang)
        self.endpoint = endpoint
        self.location = location
        
        # Initialize client
        try:
            from azure.ai.translation.text import TextTranslationClient
            from azure.core.credentials import AzureKeyCredential
            
            self.client = TextTranslationClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
        except ImportError:
            raise ImportError("Azure translation dependencies not found. Install with: pip install azure-ai-translation-text")
    
    def translate_text(self, text: str, max_retries: int = 3) -> str:
        """Translate text using Azure Translator."""
        if not text.strip():
            return ""
        
        # Split into chunks if text is too long
        max_chunk_size = 3000
        if len(text) > max_chunk_size:
            return self._translate_long_text(text, max_chunk_size)
        
        for attempt in range(max_retries):
            try:
                response = self.client.translate(
                    body=[{"text": text}],
                    to_language=[self.target_lang],
                    from_language=self.source_lang,
                    headers={"Ocp-Apim-Subscription-Region": self.location}
                )
                
                if response and len(response) > 0:
                    return response[0].translations[0].text
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Translation attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Translation failed after {max_retries} attempts: {e}")
                    raise
        
        return text  # Return original if all attempts fail
    
    def _translate_long_text(self, text: str, chunk_size: int) -> str:
        """Translate long text by splitting into chunks."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            if current_length + para_length > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        translated_chunks = []
        for chunk in tqdm(chunks, desc="Translating chunks", leave=False):
            translated = self.translate_text(chunk)
            translated_chunks.append(translated)
            time.sleep(0.5)  # Rate limiting
        
        return '\n\n'.join(translated_chunks)
    
    def test_connection(self) -> bool:
        """Test the Azure Translator connection."""
        try:
            test_text = "مرحبا"  # "Hello" in Arabic
            result = self.translate_text(test_text)
            logger.info(f"Test translation successful: {test_text} -> {result}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


class GoogleTranslator(BaseTranslator):
    """Google Cloud Translation API."""
    
    def __init__(
        self,
        credentials_path: str,
        project_id: str,
        source_lang: str = "ar",
        target_lang: str = "en"
    ):
        super().__init__("", source_lang, target_lang)  # No API key needed for Google
        self.credentials_path = credentials_path
        self.project_id = project_id
        
        # Set up credentials
        import os
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Initialize client
        try:
            from google.cloud import translate_v2 as translate
            self.client = translate.Client()
        except ImportError:
            raise ImportError("Google translation dependencies not found. Install with: pip install google-cloud-translate")
    
    def translate_text(self, text: str) -> str:
        """Translate text using Google Translate."""
        if not text.strip():
            return ""
        
        try:
            result = self.client.translate(
                text,
                target_language=self.target_lang,
                source_language=self.source_lang
            )
            return result['translatedText']
        except Exception as e:
            logger.error(f"Google translation failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test the Google Translate connection."""
        try:
            test_text = "مرحبا"  # "Hello" in Arabic
            result = self.translate_text(test_text)
            logger.info(f"Test translation successful: {test_text} -> {result}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


class Translator:
    """Main translator class that handles different translation services."""
    
    def __init__(
        self,
        service: str,
        api_key: str,
        source_lang: str = "ar",
        target_lang: str = "en",
        input_dir: str = "text_output",
        output_dir: str = "translated_output",
        **kwargs
    ):
        """
        Initialize translator.
        
        Args:
            service: Translation service ("azure" or "google")
            api_key: API key for the service
            source_lang: Source language code
            target_lang: Target language code
            input_dir: Input directory containing text files
            output_dir: Output directory for translated files
            **kwargs: Additional service-specific parameters
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Initialize the appropriate translator
        if service.lower() == "azure":
            self.translator = AzureTranslator(
                api_key=api_key,
                source_lang=source_lang,
                target_lang=target_lang,
                **kwargs
            )
        elif service.lower() == "google":
            self.translator = GoogleTranslator(
                credentials_path=kwargs.get('credentials_path', ''),
                project_id=kwargs.get('project_id', ''),
                source_lang=source_lang,
                target_lang=target_lang
            )
        else:
            raise ValueError(f"Unsupported translation service: {service}")
        
        logger.info(f"Initialized {service} translator for {source_lang} -> {target_lang}")
    
    def get_text_files(self) -> List[Path]:
        """Get list of text files in the input directory."""
        text_files = []
        for ext in ['*.txt', '*.TXT']:
            text_files.extend(self.input_dir.glob(ext))
        return sorted(text_files)
    
    def translate_file(self, file_path: Path) -> Dict[str, Any]:
        """Translate a single text file."""
        result = {
            'filename': file_path.name,
            'status': 'failed',
            'error': None,
            'output_path': None,
            'original_size': 0,
            'translated_size': 0
        }
        
        try:
            logger.debug(f"Translating file: {file_path}")
            
            # Read original text
            with open(file_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            if not original_text.strip():
                result['error'] = "Empty text file"
                return result
            
            result['original_size'] = len(original_text)
            
            # Clean text
            original_text = clean_text(original_text)
            
            # Translate text
            translated_text = self.translator.translate_text(original_text)
            
            if not translated_text.strip():
                result['error'] = "Translation returned empty text"
                return result
            
            result['translated_size'] = len(translated_text)
            
            # Create output filename
            output_filename = f"{file_path.stem}_{self.target_lang}.txt"
            output_path = self.output_dir / output_filename
            
            # Save translated text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            
            result.update({
                'status': 'success',
                'output_path': str(output_path)
            })
            
            logger.info(f"Successfully translated: {file_path.name}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error translating {file_path.name}: {e}")
        
        return result
    
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
                'processing_time': 0
            }
        
        logger.info(f"Found {len(text_files)} text files to translate")
        
        # Test connection first
        if not self.translator.test_connection():
            raise ConnectionError("Translation service connection test failed")
        
        results = {
            'total_files': len(text_files),
            'successful': 0,
            'failed': 0,
            'processed_files': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Process files with progress bar
        with tqdm(total=len(text_files), desc="Translating files", unit="file") as pbar:
            for file_path in text_files:
                result = self.translate_file(file_path)
                results['processed_files'].append(result)
                
                if result['status'] == 'success':
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                
                pbar.update(1)
                pbar.set_postfix(
                    success=results['successful'],
                    failed=results['failed']
                )
                
                # Memory cleanup and rate limiting
                memory_cleanup()
                time.sleep(0.1)  # Small delay between files
        
        results['processing_time'] = time.time() - start_time
        
        # Log summary
        logger.info(f"Translation completed in {results['processing_time']:.2f} seconds")
        logger.info(f"Successfully translated: {results['successful']} files")
        logger.info(f"Failed translations: {results['failed']} files")
        
        return results
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get statistics about translated files."""
        translated_files = list(self.output_dir.glob("*.txt"))
        
        if not translated_files:
            return {'total_files': 0}
        
        total_size = sum(f.stat().st_size for f in translated_files)
        
        return {
            'total_files': len(translated_files),
            'total_size_mb': total_size / (1024 * 1024),
            'average_size_kb': (total_size / len(translated_files)) / 1024,
            'languages': f"{self.source_lang} -> {self.target_lang}"
        }


def create_translator(service: str, **kwargs) -> Translator:
    """
    Factory function to create a translator instance.
    
    Args:
        service: Translation service name
        **kwargs: Service-specific parameters
        
    Returns:
        Translator instance
    """
    required_params = {
        'azure': ['api_key'],
        'google': ['credentials_path', 'project_id']
    }
    
    service_lower = service.lower()
    if service_lower not in required_params:
        raise ValueError(f"Unsupported service: {service}")
    
    # Check required parameters
    missing_params = []
    for param in required_params[service_lower]:
        if param not in kwargs or not kwargs[param]:
            missing_params.append(param)
    
    if missing_params:
        raise ValueError(f"Missing required parameters for {service}: {missing_params}")
    
    return Translator(service=service, **kwargs)


if __name__ == "__main__":
    # Example usage
    translator = create_translator(
        service="azure",
        api_key="your-api-key",
        input_dir="text_output",
        output_dir="translated_output"
    )
    
    results = translator.process_directory()
    print(f"Translated {results['successful']} files successfully")