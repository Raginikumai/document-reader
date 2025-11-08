"""
Pipeline Orchestration Module

This module serves as the central orchestration layer for the complete document processing pipeline.
It coordinates the flow of data through all pipeline stages, from raw document files to vector
embeddings ready for storage and retrieval.

The pipeline implements the complete data preparation workflow:
    1. Document Text Extraction (with normalization)
    2. Text Chunking (with configurable overlap)
    3. Vector Embedding Generation (semantic representation)

Usage:
    from pipeline import process_document_to_embeddings, process_document_to_chunks

    # Complete pipeline: Document → Embeddings
    embeddings = process_document_to_embeddings("sample_document.pdf")

    # Partial pipeline: Document → Chunks (if you only need text chunks)
    chunks = process_document_to_chunks("sample_document.pdf")

    # Custom parameters
    embeddings = process_document_to_embeddings(
        "sample_document.pdf",
        chunk_size=1500,
        chunk_overlap=300
    )

Architecture:
    This module embodies modular, configuration-driven design:
    - Imports specialized modules for each pipeline stage
    - Uses PipelineConfig as the single source of truth for all settings
    - Provides high-level orchestration functions for common workflows
    - Maintains clear separation of concerns
"""

import os
from pathlib import Path
from document_text_extractor import extract_text_from_document
from text_chunker import chunk_text
from embedding_generator import generate_embeddings
from config import PipelineConfig


def process_document_to_chunks(
    file_path: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> list[str]:
    """
    Process a document file through extraction and chunking stages.

    This function orchestrates the first two stages of the pipeline:
        1. Extracts and normalizes text from the document
        2. Splits the text into semantically coherent chunks
        3. Returns the chunks ready for embedding

    Configuration Priority:
        - If chunk_size or chunk_overlap are provided, they override defaults
        - Otherwise, uses values from PipelineConfig.chunking

    Args:
        file_path: Path to the document file (supports .txt, .pdf, .docx)
        chunk_size: Optional override for maximum chunk size in characters
                   (default: PipelineConfig.chunking.CHUNK_SIZE = 1000)
        chunk_overlap: Optional override for chunk overlap in characters
                      (default: PipelineConfig.chunking.CHUNK_OVERLAP = 200)

    Returns:
        A list of text chunks (strings), each respecting the chunk_size limit
        and containing overlapping content to preserve context at boundaries.

    Raises:
        Returns error string if document format is not supported
        (proper exception handling will be added in future versions)

    Example:
        >>> chunks = process_document_to_chunks("rfp_document.pdf")
        >>> print(f"Created {len(chunks)} chunks from the document")
        Created 25 chunks from the document

        >>> # Use custom chunking parameters
        >>> small_chunks = process_document_to_chunks(
        ...     "rfp_document.pdf",
        ...     chunk_size=500,
        ...     chunk_overlap=100
        ... )
    """
    # Step 1: Determine chunking configuration
    # Use provided arguments or fall back to defaults from config
    final_chunk_size = chunk_size if chunk_size is not None else PipelineConfig.chunking.CHUNK_SIZE
    final_chunk_overlap = chunk_overlap if chunk_overlap is not None else PipelineConfig.chunking.CHUNK_OVERLAP

    # Step 2: Extract and normalize text from document
    clean_text = extract_text_from_document(file_path)

    # Step 3: Check if extraction was successful (error messages start with "File type")
    if clean_text.startswith("File type"):
        # Return the error message from the extractor
        return clean_text

    # Step 4: Chunk the clean text using determined configuration
    chunks = chunk_text(
        clean_text,
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap
    )

    # Step 5: Return the final chunks
    return chunks


def process_document_to_embeddings(
    file_path: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    model_name: str = None
) -> list[list[float]]:
    """
    Process a document file through the complete data preparation pipeline.

    This function orchestrates the end-to-end workflow from document to embeddings:
        1. Extracts and normalizes text from the document
        2. Splits the text into semantically coherent chunks
        3. Generates vector embeddings for each chunk
        4. Returns embeddings ready for vector store storage

    This is the primary high-level function for complete document processing.

    Configuration Priority:
        - Custom parameters override PipelineConfig defaults
        - All settings come from PipelineConfig if not specified
        - Ensures consistent, reproducible processing

    Args:
        file_path: Path to the document file (supports .txt, .pdf, .docx)
        chunk_size: Optional override for chunk size in characters
                   (default: PipelineConfig.chunking.CHUNK_SIZE = 1000)
        chunk_overlap: Optional override for chunk overlap in characters
                      (default: PipelineConfig.chunking.CHUNK_OVERLAP = 200)
        model_name: Optional override for embedding model
                   (default: PipelineConfig.embedding.MODEL_NAME)

    Returns:
        A list of embedding vectors (list[list[float]]), where:
            - Outer list length = number of text chunks
            - Inner list length = embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
            - Each embedding represents the semantic meaning of one chunk

    Raises:
        Returns error string if document format is not supported
        (proper exception handling will be added in future versions)

    Example:
        >>> embeddings = process_document_to_embeddings("rfp_document.pdf")
        >>> print(f"Generated {len(embeddings)} embeddings")
        Generated 25 embeddings
        >>> print(f"Each embedding has {len(embeddings[0])} dimensions")
        Each embedding has 384 dimensions

        >>> # Use custom parameters
        >>> embeddings = process_document_to_embeddings(
        ...     "rfp_document.pdf",
        ...     chunk_size=1500,
        ...     chunk_overlap=300,
        ...     model_name='sentence-transformers/all-mpnet-base-v2'
        ... )

    Performance Notes:
        - First run downloads the embedding model (~80MB for all-MiniLM-L6-v2)
        - Subsequent runs use cached model
        - Processing speed: ~100-500 chunks/second on CPU
        - GPU acceleration available if configured
    """
    # Step 1: Process document to chunks using existing function
    chunks = process_document_to_chunks(
        file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Step 2: Check if chunking was successful (error messages are strings starting with "File type")
    if isinstance(chunks, str) and chunks.startswith("File type"):
        # Return the error message
        return chunks

    # Step 3: Determine embedding model to use
    final_model_name = model_name if model_name is not None else PipelineConfig.embedding.MODEL_NAME

    # Step 4: Generate embeddings for all chunks
    embeddings = generate_embeddings(chunks, final_model_name)

    # Step 5: Return the final embeddings
    return embeddings


if __name__ == '__main__':
    # Comprehensive test block demonstrating the complete end-to-end pipeline

    print("=" * 80)
    print("DOCUMENT PROCESSING PIPELINE - COMPLETE END-TO-END TEST")
    print("=" * 80)

    # Test document path (update this to test with different files)
    test_document = "sample_document.txt"

    # Check if test document exists
    if not os.path.exists(test_document):
        print(f"\n[ERROR] Test document '{test_document}' not found!")
        print("\nTo test the pipeline, please ensure you have a test document available.")
        print("Supported formats: .txt, .pdf, .docx")
        print("\n" + "=" * 80)
        exit(1)

    # Display initial file information
    print("\n--- STAGE 0: INPUT DOCUMENT ---")
    file_path_obj = Path(test_document)
    file_size = os.path.getsize(test_document)
    print(f"File: {test_document}")
    print(f"Format: {file_path_obj.suffix.upper()}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")

    # Display complete pipeline configuration
    print("\n--- PIPELINE CONFIGURATION ---")
    print(f"Extraction:")
    print(f"  - Supported formats: {', '.join(PipelineConfig.extraction.SUPPORTED_FORMATS)}")
    print(f"  - Text encoding: {PipelineConfig.extraction.TEXT_ENCODING}")
    print(f"Chunking:")
    print(f"  - Chunk size: {PipelineConfig.chunking.CHUNK_SIZE} characters")
    print(f"  - Chunk overlap: {PipelineConfig.chunking.CHUNK_OVERLAP} characters")
    print(f"  - Overlap percentage: {(PipelineConfig.chunking.CHUNK_OVERLAP / PipelineConfig.chunking.CHUNK_SIZE) * 100:.1f}%")
    print(f"Embedding:")
    print(f"  - Model: {PipelineConfig.embedding.MODEL_NAME}")
    print(f"  - Dimension: {PipelineConfig.embedding.MODEL_DIMENSION}")
    print(f"  - Device: {PipelineConfig.embedding.DEVICE}")

    # Stage 1: Text extraction and normalization
    print("\n--- STAGE 1: TEXT EXTRACTION & NORMALIZATION ---")
    print("Extracting text from document...")
    clean_text = extract_text_from_document(test_document)

    # Check for extraction errors
    if clean_text.startswith("File type"):
        print(f"[ERROR] {clean_text}")
        exit(1)

    print("[OK] Text extraction complete")
    print(f"Clean text length: {len(clean_text):,} characters")
    print(f"Word count (approx): {len(clean_text.split()):,} words")
    print(f"Line count: {clean_text.count(chr(10)) + 1:,} lines")

    # Stage 2: Text chunking
    print("\n--- STAGE 2: TEXT CHUNKING ---")
    print("Splitting text into chunks...")
    chunks = chunk_text(
        clean_text,
        chunk_size=PipelineConfig.chunking.CHUNK_SIZE,
        chunk_overlap=PipelineConfig.chunking.CHUNK_OVERLAP
    )
    print("[OK] Text chunking complete")

    # Display chunking statistics
    print("\n--- CHUNKING STATISTICS ---")
    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        chunk_lengths = [len(chunk) for chunk in chunks]
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)
        min_chunk_length = min(chunk_lengths)
        max_chunk_length = max(chunk_lengths)

        print(f"Average chunk length: {avg_chunk_length:.0f} characters")
        print(f"Minimum chunk length: {min_chunk_length} characters")
        print(f"Maximum chunk length: {max_chunk_length} characters")

        # Calculate total characters across all chunks (will be more than original due to overlap)
        total_chunk_chars = sum(chunk_lengths)
        duplication_ratio = total_chunk_chars / len(clean_text) if len(clean_text) > 0 else 0
        print(f"Total characters across all chunks: {total_chunk_chars:,}")
        print(f"Duplication ratio (due to overlap): {duplication_ratio:.2f}x")

    # Display sample chunks
    print("\n--- SAMPLE CHUNKS ---")
    chunks_to_display = min(2, len(chunks))
    for i in range(chunks_to_display):
        print(f"\nChunk {i + 1} of {len(chunks)}:")
        print(f"  Length: {len(chunks[i])} characters")
        print(f"  Preview (first 100 chars): {chunks[i][:100]}...")

    if len(chunks) > chunks_to_display:
        print(f"\n  ... and {len(chunks) - chunks_to_display} more chunks")

    # Stage 3: Vector embedding generation
    print("\n--- STAGE 3: VECTOR EMBEDDING GENERATION ---")
    print(f"Generating embeddings using model: {PipelineConfig.embedding.MODEL_NAME}")
    print("(First run will download the model - this may take a minute)")

    embeddings = generate_embeddings(chunks, PipelineConfig.embedding.MODEL_NAME)

    print("[OK] Embedding generation complete")

    # Display embedding statistics
    print("\n--- EMBEDDING STATISTICS ---")
    print(f"Total embeddings generated: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")

    if embeddings:
        # Verify dimension matches configuration
        expected_dim = PipelineConfig.embedding.MODEL_DIMENSION
        actual_dim = len(embeddings[0])
        if actual_dim == expected_dim:
            print(f"[OK] Dimension matches configuration ({expected_dim})")
        else:
            print(f"[WARNING] Dimension {actual_dim} differs from configured {expected_dim}")

        # Calculate vector norms to verify normalization
        import math
        norms = [math.sqrt(sum(x * x for x in emb)) for emb in embeddings[:3]]
        avg_norm = sum(norms) / len(norms)
        print(f"Average vector norm (first 3): {avg_norm:.6f}")
        print(f"Embeddings are normalized: {'Yes' if 0.99 < avg_norm < 1.01 else 'No'}")

    # Display sample embedding
    print("\n--- SAMPLE EMBEDDING ---")
    if embeddings:
        print(f"First embedding (first 10 values):")
        print(f"  {embeddings[0][:10]}")
        print(f"  ... ({len(embeddings[0]) - 10} more values)")

    # Calculate memory footprint
    if embeddings:
        # Each float is typically 8 bytes (64-bit)
        total_floats = len(embeddings) * len(embeddings[0])
        memory_bytes = total_floats * 8
        memory_mb = memory_bytes / (1024 * 1024)
        print("\n--- MEMORY FOOTPRINT ---")
        print(f"Total embedding values: {total_floats:,}")
        print(f"Estimated memory usage: {memory_mb:.2f} MB")

    # Complete pipeline summary
    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"[OK] Document processed successfully: {test_document}")
    print(f"[OK] Stage 1 - Extraction: {len(clean_text):,} characters")
    print(f"[OK] Stage 2 - Chunking: {len(chunks)} chunks")
    print(f"[OK] Stage 3 - Embedding: {len(embeddings)} vectors ({len(embeddings[0])}D)")
    print(f"[OK] Data preparation complete!")
    print("=" * 80)

    # Demonstrate the high-level end-to-end function
    print("\n\n" + "=" * 80)
    print("DEMONSTRATION: HIGH-LEVEL END-TO-END FUNCTION")
    print("=" * 80)
    print("\nUsing process_document_to_embeddings() for complete pipeline...")

    embeddings_e2e = process_document_to_embeddings(test_document)

    print(f"\n[OK] Complete pipeline executed successfully!")
    print(f"  - Input: {test_document}")
    print(f"  - Output: {len(embeddings_e2e)} embeddings of dimension {len(embeddings_e2e[0])}")
    print(f"  - Single function call handled all stages automatically")

    # Test with custom parameters
    print("\n\n" + "=" * 80)
    print("DEMONSTRATION: CUSTOM PARAMETERS")
    print("=" * 80)
    print("\nProcessing same document with custom chunk settings...")
    print("Custom settings: chunk_size=500, chunk_overlap=100")

    custom_embeddings = process_document_to_embeddings(
        test_document,
        chunk_size=500,
        chunk_overlap=100
    )

    print(f"\nResults with custom parameters:")
    print(f"  - Total embeddings: {len(custom_embeddings)}")
    print(f"  - Embedding dimension: {len(custom_embeddings[0])}")
    print(f"\nComparison:")
    print(f"  - Default settings: {len(embeddings)} embeddings")
    print(f"  - Custom settings: {len(custom_embeddings)} embeddings")
    print(f"  - Difference: {abs(len(custom_embeddings) - len(embeddings))} embeddings")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE - PIPELINE READY FOR PRODUCTION!")
    print("=" * 80)

    print("\nData Preparation Pipeline Status: COMPLETE")
    print("  [OK] Document extraction and normalization")
    print("  [OK] Intelligent text chunking")
    print("  [OK] Vector embedding generation")
    print("\nNext steps:")
    print("  1. Implement vector store for embedding storage")
    print("  2. Add query and retrieval capabilities")
    print("  3. Integrate with LLM for JSON output generation")
    print("  4. Build end-user interface or API")
