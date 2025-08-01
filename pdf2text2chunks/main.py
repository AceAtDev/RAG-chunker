"""
PDF to RAG Converter
A clean, modular tool for converting PDFs to text and creating RAG-ready chunks.

Usage:
    python main.py --help
    python main.py convert --input pdfs/ --output text/
    python main.py chunk --input text/ --output chunks/ --azure-key YOUR_KEY
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

from src.config import Config
from src.pdf_converter import PDFConverter
from src.text_chunker import TextChunker
from src.translator import Translator
from src.utils import setup_logging, validate_directories

__version__ = "1.0.0"


def setup_cli():
    """Setup command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert PDFs to text and create RAG-ready chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
Examples:
  # Convert PDFs to text
  python main.py convert --input pdfs/ --output text/
  
  # Create chunks using config.json credentials
  python main.py chunk --input text/ --output chunks/
  
  # Create chunks with custom API key (overrides config.json)
  python main.py chunk --input text/ --output chunks/ --azure-key YOUR_KEY
  
  # Create chunks with embeddings only
  python main.py chunk --input text/ --output chunks/ --no-bm25
  
  # Create chunks with BM25 only (no embeddings)
  python main.py chunk --input text/ --output chunks/ --bm25-only
  
  # Full pipeline using config.json credentials
  python main.py pipeline --input pdfs/
  
  # Full pipeline with custom credentials
  python main.py pipeline --input pdfs/ --azure-key YOUR_KEY --embeddings-only
        """
    )
    
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--config", help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert PDFs to text")
    convert_parser.add_argument("--input", "-i", required=True, help="Input directory containing PDFs")
    convert_parser.add_argument("--output", "-o", required=True, help="Output directory for text files")
    convert_parser.add_argument("--method", choices=["standard", "ocr"], default="standard", 
                              help="Conversion method (standard for normal PDFs, ocr for scanned/Arabic)")
    convert_parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")
    
    # Chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Create chunks with embeddings and/or BM25")
    chunk_parser.add_argument("--input", "-i", required=True, help="Input directory containing text files")
    chunk_parser.add_argument("--output", "-o", required=True, help="Output directory for chunk files")
    chunk_parser.add_argument("--azure-key", help="Azure OpenAI API key (optional, uses config.json if not provided)")
    chunk_parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint (optional, uses config.json if not provided)")
    chunk_parser.add_argument("--chunk-size", type=int, default=512, help="Maximum chunk size in tokens")
    chunk_parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks")

    # Vector generation options
    vector_group = chunk_parser.add_mutually_exclusive_group()
    vector_group.add_argument("--embeddings-only", action="store_true", 
                            help="Generate embeddings only (no BM25)")
    vector_group.add_argument("--bm25-only", action="store_true", 
                            help="Generate BM25 sparse vectors only (no embeddings)")
    vector_group.add_argument("--no-bm25", action="store_true", 
                            help="Disable BM25 generation (embeddings only)")
    
    chunk_parser.add_argument("--bm25-max-dim", type=int, default=1000, 
                            help="Maximum dimensions for BM25 sparse vectors")
    
    # Translate command
    translate_parser = subparsers.add_parser("translate", help="Translate text files")
    translate_parser.add_argument("--input", "-i", required=True, help="Input directory containing text files")
    translate_parser.add_argument("--output", "-o", required=True, help="Output directory for translated files")
    translate_parser.add_argument("--translator", choices=["azure", "google"], default="azure", 
                                help="Translation service to use")
    translate_parser.add_argument("--key", required=True, help="API key for translation service")
    translate_parser.add_argument("--source-lang", default="ar", help="Source language code")
    translate_parser.add_argument("--target-lang", default="en", help="Target language code")
    
    # Pipeline command (full workflow)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full PDF to chunks pipeline")
    pipeline_parser.add_argument("--input", "-i", required=True, help="Input directory containing PDFs")
    pipeline_parser.add_argument("--output", "-o", default="output", help="Base output directory")
    pipeline_parser.add_argument("--azure-key", help="Azure OpenAI API key (optional, uses config.json if not provided)")
    pipeline_parser.add_argument("--convert-method", choices=["standard", "ocr"], default="standard")
    pipeline_parser.add_argument("--translate", action="store_true", help="Include translation step")
    pipeline_parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers")

    # Vector options for pipeline
    pipeline_vector_group = pipeline_parser.add_mutually_exclusive_group()
    pipeline_vector_group.add_argument("--embeddings-only", action="store_true", 
                                     help="Generate embeddings only (no BM25)")
    pipeline_vector_group.add_argument("--bm25-only", action="store_true", 
                                     help="Generate BM25 sparse vectors only (no embeddings)")
    
    return parser


def get_vector_settings(args, config: Config) -> tuple:
    """
    Determine vector generation settings based on arguments and config.
    
    Returns:
        tuple: (enable_embeddings, enable_bm25)
    """
    # Start with config defaults
    enable_embeddings = config.vectors.enable_embeddings
    enable_bm25 = config.vectors.enable_bm25
    
    # Override based on command line arguments
    if hasattr(args, 'embeddings_only') and args.embeddings_only:
        enable_embeddings = True
        enable_bm25 = False
    elif hasattr(args, 'bm25_only') and args.bm25_only:
        enable_embeddings = False
        enable_bm25 = True
    elif hasattr(args, 'no_bm25') and args.no_bm25:
        enable_embeddings = True
        enable_bm25 = False
    
    return enable_embeddings, enable_bm25


def validate_vector_settings(enable_embeddings: bool, enable_bm25: bool, azure_key: str = None):
    """Validate vector generation settings."""
    if not enable_embeddings and not enable_bm25:
        raise ValueError("At least one of embeddings or BM25 must be enabled")
    
    if enable_embeddings and not azure_key:
        raise ValueError("Azure OpenAI API key is required when embeddings are enabled. Please provide --azure-key or set it in config.json")


def convert_command(args):
    """Handle PDF conversion command."""
    logger = logging.getLogger(__name__)
    logger.info(f"Converting PDFs from {args.input} to {args.output}")
    
    # Validate directories
    validate_directories(input_dir=args.input, create_output=args.output)
    
    # Initialize converter
    config = Config(args.config if hasattr(args, 'config') else None)
    if args.method == "ocr":
        # Use OCR converter for scanned/Arabic PDFs
        from src.ocr_converter import OCRConverter
        converter = OCRConverter(
            output_dir=args.output,
            max_workers=args.max_workers
        )
    else:
        # Use standard converter
        converter = PDFConverter(
            input_dir=args.input,
            output_dir=args.output
        )
    
    # Process files
    results = converter.process_directory()
    
    # Print results
    logger.info("Conversion completed!")
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    
    return results


def chunk_command(args):
    """Handle text chunking command."""
    logger = logging.getLogger(__name__)
    logger.info(f"Creating chunks from {args.input} to {args.output}")
    
    # Validate directories
    validate_directories(input_dir=args.input, create_output=args.output)
    
    # Load configuration
    config = Config(args.config if hasattr(args, 'config') else None)
    
    # Determine vector settings
    enable_embeddings, enable_bm25 = get_vector_settings(args, config)
    
    # Get Azure credentials (prioritize command line, fallback to config)
    azure_key = args.azure_key or config.azure.api_key
    azure_endpoint = getattr(args, 'azure_endpoint', None) or config.azure.endpoint
    
    # Validate settings
    validate_vector_settings(enable_embeddings, enable_bm25, azure_key)
    
    # Log configuration source
    if args.azure_key:
        logger.info("Using Azure OpenAI API key from command line")
    elif config.azure.api_key:
        logger.info("Using Azure OpenAI API key from config.json")
    
    if azure_endpoint:
        if getattr(args, 'azure_endpoint', None):
            logger.info("Using Azure OpenAI endpoint from command line")
        else:
            logger.info("Using Azure OpenAI endpoint from config.json")
    
    # Log vector generation mode
    if enable_embeddings and enable_bm25:
        mode = "embeddings + BM25 sparse vectors"
    elif enable_embeddings:
        mode = "embeddings only"
    else:
        mode = "BM25 sparse vectors only"
    
    logger.info(f"Vector generation mode: {mode}")
    
    # Initialize chunker
    chunker = TextChunker(
        input_dir=args.input,
        output_dir=args.output,
        azure_api_key=azure_key if enable_embeddings else None,
        azure_endpoint=azure_endpoint,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        enable_embeddings=enable_embeddings,
        enable_bm25=enable_bm25,
        bm25_max_dim=getattr(args, 'bm25_max_dim', config.vectors.bm25_max_dim)
    )
    
    # Process files
    results = chunker.process_directory()
    
    # Print results
    logger.info("Chunking completed!")
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Total chunks: {results['total_chunks']}")
    
    return results


def translate_command(args):
    """Handle translation command."""
    logger = logging.getLogger(__name__)
    logger.info(f"Translating files from {args.input} to {args.output}")
    
    # Validate directories
    validate_directories(input_dir=args.input, create_output=args.output)
    
    # Initialize translator
    translator = Translator(
        service=args.translator,
        api_key=args.key,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        input_dir=args.input,
        output_dir=args.output
    )
    
    # Process files
    results = translator.process_directory()
    
    # Print results
    logger.info("Translation completed!")
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    
    return results


def pipeline_command(args):
    """Handle full pipeline command."""
    logger = logging.getLogger(__name__)
    logger.info("Running full PDF to RAG chunks pipeline")
    
    # Load configuration
    config = Config(args.config if hasattr(args, 'config') else None)
    
    # Determine vector settings
    enable_embeddings, enable_bm25 = get_vector_settings(args, config)
    
    # Get Azure credentials (prioritize command line, fallback to config)
    azure_key = args.azure_key or config.azure.api_key
    
    # Validate settings
    validate_vector_settings(enable_embeddings, enable_bm25, azure_key)
    
    # Log configuration source
    if args.azure_key:
        logger.info("Using Azure OpenAI API key from command line")
    elif config.azure.api_key:
        logger.info("Using Azure OpenAI API key from config.json")
    
    # Setup output directories
    base_output = Path(args.output)
    text_dir = base_output / "text"
    chunks_dir = base_output / "chunks"
    
    if args.translate:
        translated_dir = base_output / "translated"
        translated_dir.mkdir(parents=True, exist_ok=True)
    
    text_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert PDFs to text
    logger.info("Step 1: Converting PDFs to text...")
    convert_args = argparse.Namespace(
        input=args.input,
        output=str(text_dir),
        method=args.convert_method,
        max_workers=args.max_workers,
        config=getattr(args, 'config', None)
    )
    convert_results = convert_command(convert_args)
    
    if convert_results['successful'] == 0:
        logger.error("No PDFs were successfully converted. Stopping pipeline.")
        return {"error": "No successful conversions"}
    
    # Step 2: Translation (optional)
    input_for_chunking = str(text_dir)
    translate_results = None
    if args.translate:
        logger.info("Step 2: Translating text files...")
        translate_args = argparse.Namespace(
            input=str(text_dir),
            output=str(translated_dir),
            translator="azure",  # Default to Azure
            key=args.azure_key,
            source_lang="ar",
            target_lang="en"
        )
        translate_results = translate_command(translate_args)
        input_for_chunking = str(translated_dir)
    
    # Step 3: Create chunks
    logger.info("Step 3: Creating chunks...")
    
    # Log vector generation mode
    if enable_embeddings and enable_bm25:
        mode = "embeddings + BM25 sparse vectors"
    elif enable_embeddings:
        mode = "embeddings only"
    else:
        mode = "BM25 sparse vectors only"
    
    logger.info(f"Vector generation mode: {mode}")
    
    chunk_args = argparse.Namespace(
        input=input_for_chunking,
        output=str(chunks_dir),
        azure_key=azure_key if enable_embeddings else None,
        azure_endpoint=config.azure.endpoint,
        chunk_size=512,
        chunk_overlap=50,
        embeddings_only=enable_embeddings and not enable_bm25,
        bm25_only=enable_bm25 and not enable_embeddings,
        bm25_max_dim=config.vectors.bm25_max_dim,
        config=getattr(args, 'config', None)
    )
    chunk_results = chunk_command(chunk_args)
    
    # Summary
    logger.info("Pipeline completed!")
    logger.info(f"Text files created: {convert_results['successful']}")
    if args.translate:
        logger.info(f"Files translated: {translate_results['successful']}")
    logger.info(f"Chunk files created: {chunk_results['successful']}")
    logger.info(f"Total chunks generated: {chunk_results['total_chunks']}")
    
    return {
        "convert": convert_results,
        "translate": translate_results,
        "chunk": chunk_results
    }


def main():
    """Main entry point."""
    parser = setup_cli()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Route to appropriate command handler
        if args.command == "convert":
            convert_command(args)
        elif args.command == "chunk":
            chunk_command(args)
        elif args.command == "translate":
            translate_command(args)
        elif args.command == "pipeline":
            pipeline_command(args)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())