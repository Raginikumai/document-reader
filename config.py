"""
Pipeline Configuration Module

This module serves as the single source of truth for all configuration parameters
used throughout the document processing pipeline. It provides centralized, organized
settings that can be easily modified without changing implementation code.

Usage:
    from config import PipelineConfig

    # Access configuration values
    chunk_size = PipelineConfig.chunking.CHUNK_SIZE
    chunk_overlap = PipelineConfig.chunking.CHUNK_OVERLAP

Design Philosophy:
    - Centralized: All configuration in one place
    - Organized: Grouped by pipeline stage (chunking, embedding, etc.)
    - Documented: Each setting includes explanation and reasoning
    - Extensible: Easy to add new configuration sections as pipeline grows
"""


class ChunkingConfig:
    """
    Configuration settings for the text chunking stage of the pipeline.

    Text chunking splits long documents into smaller, semantically coherent segments
    that are appropriately sized for vector embedding models and retrieval operations.

    Attributes:
        CHUNK_SIZE: Maximum size of each text chunk in characters.
                   Default: 1000 characters
                   Reasoning:
                   - Large enough to preserve context and semantic meaning
                   - Small enough for precise retrieval of relevant information
                   - Fits well within embedding model token limits (typically 512-2048 tokens)
                   - Suitable for business documents with multi-sentence requirements

        CHUNK_OVERLAP: Number of characters to overlap between consecutive chunks.
                      Default: 200 characters (20% of CHUNK_SIZE)
                      Reasoning:
                      - Ensures concepts split across chunk boundaries are preserved
                      - Provides context continuity between adjacent chunks
                      - Improves retrieval quality by reducing edge-case losses
                      - Not so large as to create excessive duplication

    Notes:
        - These values are starting recommendations based on common best practices
        - Optimal values may vary depending on:
          * Document type and structure (RFPs, specifications, catalogs, etc.)
          * Embedding model being used
          * Retrieval precision requirements
        - Consider experimentation with different values for your specific use case
        - Overlap should generally be 10-25% of chunk size
    """

    # Maximum size of each text chunk in characters
    CHUNK_SIZE: int = 1000

    # Number of characters to overlap between consecutive chunks
    CHUNK_OVERLAP: int = 200

    # Separator hierarchy for RecursiveCharacterTextSplitter
    # Priority order: paragraphs → sentences → words → characters
    # Note: This is used by the text_chunker module
    SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


class ExtractionConfig:
    """
    Configuration settings for the document text extraction stage.

    This stage handles reading various document formats and normalizing the extracted text.

    Attributes:
        SUPPORTED_FORMATS: List of file extensions that the pipeline can process.
        TEXT_ENCODING: Character encoding to use for plain text files.

    Notes:
        - Additional format support can be added by updating SUPPORTED_FORMATS
          and implementing the corresponding loader in document_text_extractor.py
    """

    # Supported document file formats
    SUPPORTED_FORMATS: list[str] = ['.txt', '.pdf', '.docx']

    # Character encoding for text files (UTF-8 is standard for modern text)
    TEXT_ENCODING: str = 'utf-8'


class EmbeddingConfig:
    """
    Configuration settings for the vector embedding generation stage.

    Vector embeddings transform text chunks into dense numerical representations
    that capture semantic meaning. These embeddings enable similarity-based search
    and are essential for building retrieval-augmented generation (RAG) systems.

    Attributes:
        MODEL_NAME: HuggingFace model identifier for the embedding model.
                   Default: 'sentence-transformers/all-MiniLM-L6-v2'
                   Reasoning:
                   - Industry-standard model for semantic search
                   - Excellent balance of quality, speed, and size (~80MB)
                   - 384-dimensional embeddings (compact yet effective)
                   - Optimized for retrieval tasks
                   - Well-supported and widely used in production

        MODEL_DIMENSION: Size of the embedding vectors produced by the model.
                        Default: 384 (for all-MiniLM-L6-v2)
                        Reasoning:
                        - Must match the model's output dimension
                        - Used by vector stores to configure index structure
                        - Smaller dimensions = faster search, less storage
                        - 384 dimensions provide good semantic representation

        DEVICE: Computation device for embedding generation.
               Default: 'cpu'
               Options: 'cpu', 'cuda' (NVIDIA GPU), 'mps' (Apple M1/M2)
               Reasoning:
               - CPU works universally without special setup
               - Can be changed to 'cuda' for GPU acceleration if available
               - GPU can provide 10-100x speed improvement for large batches

        BATCH_SIZE: Number of text chunks to process in a single batch.
                   Default: 32
                   Reasoning:
                   - Batching improves efficiency vs. one-at-a-time processing
                   - 32 is safe for CPU with moderate memory usage
                   - Can be increased to 128+ when using GPU
                   - Larger batches = faster but more memory

        NORMALIZE_EMBEDDINGS: Whether to normalize embedding vectors to unit length.
                             Default: True
                             Reasoning:
                             - Normalized vectors enable cosine similarity via dot product
                             - Simplifies similarity calculations (no division needed)
                             - Standard practice for retrieval systems
                             - Recommended by sentence-transformers library

        SHOW_PROGRESS: Whether to display a progress bar during embedding generation.
                      Default: True
                      Reasoning:
                      - Helpful for monitoring long-running operations
                      - Provides user feedback for large document sets
                      - Can be disabled for production/automated workflows

    Notes:
        - The model will be automatically downloaded on first use (~80MB)
        - Models are cached locally in ~/.cache/huggingface/
        - Processing speed: ~100-500 chunks/second on CPU (model-dependent)
        - GPU acceleration can improve speed 10-100x for large batches
    """

    # HuggingFace model identifier
    # This model is downloaded automatically on first use and cached locally
    MODEL_NAME: str = 'sentence-transformers/all-MiniLM-L6-v2'

    # Embedding vector dimension (must match the model's output)
    # Vector stores need this value to configure their index structure
    MODEL_DIMENSION: int = 384

    # Device for computation: 'cpu', 'cuda' (GPU), or 'mps' (Mac M1/M2)
    # Default to CPU for universal compatibility
    DEVICE: str = 'cpu'

    # Batch processing size - number of chunks to embed at once
    # Larger batches are faster but use more memory
    # Recommended: 32 for CPU, 128+ for GPU
    BATCH_SIZE: int = 32

    # Normalize embeddings to unit length (recommended for cosine similarity)
    # This is standard practice for retrieval systems
    NORMALIZE_EMBEDDINGS: bool = True

    # Show progress bar during embedding generation (useful for large batches)
    # Disable this for production/automated workflows
    SHOW_PROGRESS: bool = True


class VectorStoreConfig:
    """
    Configuration settings for the vector store operations.

    Vector stores enable efficient similarity search over embedded documents. This stage
    stores text chunks and their embeddings, making them searchable for retrieval-augmented
    generation (RAG) applications.

    Attributes:
        STORE_TYPE: Type of vector store implementation to use.
                   Default: 'faiss'
                   Reasoning:
                   - FAISS is optimized for fast similarity search
                   - Works locally without external dependencies
                   - Supports both CPU and GPU acceleration
                   - Easy to persist and load from disk
                   - Perfect for our current scale (100s-1000s of documents)

        INDEX_TYPE: FAISS index type for similarity search.
                   Default: 'IndexFlatL2'
                   Options:
                   - 'IndexFlatL2': Exact search using L2 distance (Euclidean)
                   - 'IndexFlatIP': Exact search using inner product (dot product)
                   - 'IndexIVFFlat': Approximate search (faster for large datasets)
                   Reasoning:
                   - IndexFlatL2 provides exact search with best quality
                   - Suitable for datasets up to ~1M vectors
                   - L2 distance works well with normalized embeddings
                   - Can upgrade to IVF for larger datasets later

        DEFAULT_SEARCH_K: Default number of similar results to return from searches.
                         Default: 5
                         Reasoning:
                         - 5 results provide good context without overwhelming the LLM
                         - Balances relevance (more is better) vs. noise (less is better)
                         - Can be overridden per query for flexibility
                         - Common default in RAG systems

        PERSIST_DIRECTORY: Directory path for saving and loading vector stores.
                          Default: 'vector_stores'
                          Reasoning:
                          - Centralized location for all vector store files
                          - Relative path works in development and production
                          - Easy to backup and version control (though stores can be large)
                          - Subdirectories can organize multiple stores by project/document type

        SAVE_FORMAT: File format for vector store persistence.
                    Default: 'local'
                    Reasoning:
                    - 'local' uses FAISS native format (index.faiss + index.pkl)
                    - Fast save and load operations
                    - Preserves all metadata and index structure
                    - Compatible across platforms (Windows, Linux, Mac)

        SIMILARITY_METRIC: Distance metric for similarity calculations.
                          Default: 'l2'
                          Options: 'l2' (Euclidean), 'cosine', 'inner_product'
                          Reasoning:
                          - L2 distance works well with normalized embeddings
                          - Cosine similarity is equivalent to L2 for normalized vectors
                          - Consistent with sentence-transformers recommendations

    Notes:
        - Vector stores are saved as directories containing multiple files
        - Each store requires ~8 bytes per dimension per chunk for storage
        - For 1000 chunks with 384D embeddings: ~3MB storage
        - Loading a store is fast (typically <100ms for 1000 chunks)
        - Can have multiple stores for different document collections
    """

    # Type of vector store to use ('faiss', 'chroma', 'pinecone', etc.)
    # Starting with FAISS for local, fast, simple implementation
    STORE_TYPE: str = 'faiss'

    # FAISS index type - determines search algorithm
    # IndexFlatL2 = exact search, best quality, good for <1M vectors
    INDEX_TYPE: str = 'IndexFlatL2'

    # Default number of similar chunks to return in searches
    # Can be overridden per query
    DEFAULT_SEARCH_K: int = 5

    # Directory where vector stores are saved and loaded
    # Each store is a subdirectory within this path
    PERSIST_DIRECTORY: str = 'vector_stores'

    # File format for saving stores
    # 'local' uses FAISS native format (fastest, most compatible)
    SAVE_FORMAT: str = 'local'

    # Similarity metric for distance calculations
    # 'l2' = Euclidean distance, works well with normalized embeddings
    SIMILARITY_METRIC: str = 'l2'

    # Whether to automatically save stores after creation
    # Useful for ensuring stores are persisted without explicit save calls
    AUTO_SAVE: bool = False

    # Store naming convention: 'timestamp', 'incremental', 'custom'
    # Helps organize multiple stores from different runs
    NAMING_CONVENTION: str = 'custom'


class PipelineConfig:
    """
    Master configuration object for the entire document processing pipeline.

    This class aggregates all configuration sections, providing a single entry point
    for accessing any pipeline setting.

    Attributes:
        extraction: Configuration for document extraction and normalization
        chunking: Configuration for text chunking operations
        embedding: Configuration for vector embedding generation
        vector_store: Configuration for vector store operations

    Usage:
        from config import PipelineConfig

        # Access specific configuration values
        max_chunk_size = PipelineConfig.chunking.CHUNK_SIZE
        supported_formats = PipelineConfig.extraction.SUPPORTED_FORMATS
        embedding_model = PipelineConfig.embedding.MODEL_NAME
        search_k = PipelineConfig.vector_store.DEFAULT_SEARCH_K

    Future Expansion:
        As new pipeline stages are added, new configuration sections will be added here:
        - llm: Configuration for LLM integration and prompting
        - output: Configuration for JSON output generation
    """

    # Document extraction and normalization settings
    extraction = ExtractionConfig()

    # Text chunking settings
    chunking = ChunkingConfig()

    # Vector embedding generation settings
    embedding = EmbeddingConfig()

    # Vector store operations settings
    vector_store = VectorStoreConfig()

    # Future configuration sections will be added here as the pipeline grows:
    # llm = LLMConfig()                  # Coming next
    # output = OutputConfig()            # Coming later


# Convenience function to display current configuration
def print_config():
    """
    Print the current pipeline configuration in a human-readable format.

    Useful for debugging and verification of configuration settings.
    """
    print("=" * 80)
    print("DOCUMENT PROCESSING PIPELINE - CONFIGURATION")
    print("=" * 80)

    print("\n--- EXTRACTION CONFIGURATION ---")
    print(f"Supported formats: {', '.join(PipelineConfig.extraction.SUPPORTED_FORMATS)}")
    print(f"Text encoding: {PipelineConfig.extraction.TEXT_ENCODING}")

    print("\n--- CHUNKING CONFIGURATION ---")
    print(f"Chunk size: {PipelineConfig.chunking.CHUNK_SIZE} characters")
    print(f"Chunk overlap: {PipelineConfig.chunking.CHUNK_OVERLAP} characters")
    print(f"Overlap percentage: {(PipelineConfig.chunking.CHUNK_OVERLAP / PipelineConfig.chunking.CHUNK_SIZE) * 100:.1f}%")
    print(f"Separator hierarchy: {PipelineConfig.chunking.SEPARATORS}")

    print("\n--- EMBEDDING CONFIGURATION ---")
    print(f"Model: {PipelineConfig.embedding.MODEL_NAME}")
    print(f"Embedding dimension: {PipelineConfig.embedding.MODEL_DIMENSION}")
    print(f"Device: {PipelineConfig.embedding.DEVICE}")
    print(f"Batch size: {PipelineConfig.embedding.BATCH_SIZE}")
    print(f"Normalize embeddings: {PipelineConfig.embedding.NORMALIZE_EMBEDDINGS}")
    print(f"Show progress: {PipelineConfig.embedding.SHOW_PROGRESS}")

    print("\n--- VECTOR STORE CONFIGURATION ---")
    print(f"Store type: {PipelineConfig.vector_store.STORE_TYPE}")
    print(f"Index type: {PipelineConfig.vector_store.INDEX_TYPE}")
    print(f"Default search K: {PipelineConfig.vector_store.DEFAULT_SEARCH_K}")
    print(f"Persist directory: {PipelineConfig.vector_store.PERSIST_DIRECTORY}")
    print(f"Save format: {PipelineConfig.vector_store.SAVE_FORMAT}")
    print(f"Similarity metric: {PipelineConfig.vector_store.SIMILARITY_METRIC}")
    print(f"Auto-save: {PipelineConfig.vector_store.AUTO_SAVE}")
    print(f"Naming convention: {PipelineConfig.vector_store.NAMING_CONVENTION}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # When run directly, display the current configuration
    print_config()

    print("\nConfiguration loaded successfully!")
    print("\nTo use in other modules:")
    print("  from config import PipelineConfig")
    print("  chunk_size = PipelineConfig.chunking.CHUNK_SIZE")
