"""PDF to text conversion with support for standard and OCR methods."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from PyPDF2 import PdfReader
from tqdm import tqdm

from .utils import clean_text, memory_cleanup


logger = logging.getLogger(__name__)


class PDFConverter:
    """Standard PDF to text converter using PyPDF2."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the PDF converter.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory where text files will be saved
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure input directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    def convert_pdf_to_text(self, pdf_path: Path) -> Optional[str]:
        """
        Convert a single PDF file to text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF, or None if conversion fails
        """
        try:
            logger.debug(f"Converting PDF: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
                
                if total_pages == 0:
                    logger.warning(f"PDF has no pages: {pdf_path}")
                    return None
                
                text_parts = []
                
                for page_num in range(total_pages):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        if page_text.strip():
                            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                        
                        # Memory cleanup for large PDFs
                        if page_num % 50 == 0:
                            memory_cleanup()
                            
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {pdf_path}: {e}")
                        continue
                
                if not text_parts:
                    logger.warning(f"No text extracted from {pdf_path}")
                    return None
                
                full_text = "\n\n".join(text_parts)
                return clean_text(full_text)
                
        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {e}")
            return None
    
    def save_text_to_file(self, text: str, output_path: Path) -> bool:
        """
        Save extracted text to a file.
        
        Args:
            text: Text to save
            output_path: Path where to save the text file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return True
        except Exception as e:
            logger.error(f"Error saving to {output_path}: {e}")
            return False
    
    def process_single_pdf(self, pdf_path: Path) -> Dict[str, any]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'filename': pdf_path.name,
            'status': 'failed',
            'error': None,
            'output_path': None
        }
        
        try:
            # Convert PDF to text
            text = self.convert_pdf_to_text(pdf_path)
            
            if text:
                # Create output path
                output_path = self.output_dir / f"{pdf_path.stem}.txt"
                
                # Save text file
                if self.save_text_to_file(text, output_path):
                    result['status'] = 'success'
                    result['output_path'] = str(output_path)
                    logger.info(f"Successfully converted: {pdf_path.name}")
                else:
                    result['error'] = "Failed to save text file"
            else:
                result['error'] = "No text extracted from PDF"
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to process {pdf_path.name}: {e}")
        
        return result
    
    def get_pdf_files(self) -> List[Path]:
        """Get list of PDF files in the input directory."""
        pdf_files = []
        for ext in ['*.pdf', '*.PDF']:
            pdf_files.extend(self.input_dir.glob(ext))
        return sorted(pdf_files)
    
    def process_directory(self, max_workers: int = 4) -> Dict[str, any]:
        """
        Process all PDF files in the input directory.
        
        Args:
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary with processing summary
        """
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'processed_files': [],
                'processing_time': 0
            }
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {
            'total_files': len(pdf_files),
            'successful': 0,
            'failed': 0,
            'processed_files': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Process files with progress bar
        with tqdm(total=len(pdf_files), desc="Converting PDFs", unit="file") as pbar:
            if max_workers == 1:
                # Sequential processing
                for pdf_path in pdf_files:
                    result = self.process_single_pdf(pdf_path)
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
            else:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_pdf = {
                        executor.submit(self.process_single_pdf, pdf_path): pdf_path
                        for pdf_path in pdf_files
                    }
                    
                    # Process completed tasks
                    for future in as_completed(future_to_pdf):
                        result = future.result()
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
        
        results['processing_time'] = time.time() - start_time
        
        # Log summary
        logger.info(f"Processing completed in {results['processing_time']:.2f} seconds")
        logger.info(f"Successfully converted: {results['successful']} files")
        logger.info(f"Failed conversions: {results['failed']} files")
        
        return results
    
    def get_pdf_info(self, pdf_path: Path) -> Dict[str, any]:
        """
        Get information about a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                metadata = reader.metadata or {}
                
                return {
                    'filename': pdf_path.name,
                    'file_size': pdf_path.stat().st_size,
                    'page_count': len(reader.pages),
                    'title': metadata.get('/Title', ''),
                    'author': metadata.get('/Author', ''),
                    'subject': metadata.get('/Subject', ''),
                    'creator': metadata.get('/Creator', ''),
                    'producer': metadata.get('/Producer', ''),
                    'creation_date': str(metadata.get('/CreationDate', '')),
                    'modification_date': str(metadata.get('/ModDate', ''))
                }
        except Exception as e:
            logger.error(f"Error getting PDF info for {pdf_path}: {e}")
            return {
                'filename': pdf_path.name,
                'error': str(e)
            }


class PDFConverterWithRetry(PDFConverter):
    """PDF converter with retry logic for failed conversions."""
    
    def __init__(self, input_dir: str, output_dir: str, max_retries: int = 3):
        super().__init__(input_dir, output_dir)
        self.max_retries = max_retries
    
    def process_single_pdf(self, pdf_path: Path) -> Dict[str, any]:
        """Process a single PDF with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = super().process_single_pdf(pdf_path)
                
                if result['status'] == 'success':
                    return result
                
                last_error = result['error']
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {pdf_path.name}: {result['error']}. Retrying...")
                    time.sleep(1)  # Brief delay before retry
                
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {pdf_path.name}: {e}. Retrying...")
                    time.sleep(1)
        
        # All attempts failed
        return {
            'filename': pdf_path.name,
            'status': 'failed',
            'error': f"Failed after {self.max_retries} attempts. Last error: {last_error}",
            'output_path': None
        }