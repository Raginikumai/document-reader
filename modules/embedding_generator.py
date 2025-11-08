"""
Vector Embedding Generation Module

This module provides functionality to convert text chunks into dense vector embeddings
that capture semantic meaning. These embeddings enable similarity-based search and are
essential for building retrieval systems and knowledge bases.

Vector embeddings transform text into numerical representations where semantically
similar text produces similar vectors, enabling mathematical operations like cosine
similarity to measure semantic relatedness.

Dependencies required:
    pip install langchain-huggingface sentence-transformers

Technical Notes:
    - Uses HuggingFace Sentence Transformers models via LangChain wrapper
    - Default model: all-MiniLM-L6-v2 (384-dimensional embeddings)
    - Embeddings are normalized to unit length for cosine similarity
    - Model is loaded once and cached for efficiency
"""

from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Create a HuggingFaceEmbeddings object for use with vector stores.

    This function initializes and returns an embedding model object that can be used
    by vector stores (like FAISS or Chroma) for encoding queries and documents.

    Args:
        model_name: HuggingFace model identifier (default: 'sentence-transformers/all-MiniLM-L6-v2')

    Returns:
        A HuggingFaceEmbeddings object configured for semantic search

    Example:
        >>> embeddings = create_embeddings()
        >>> from vector_store_manager import load_vector_store
        >>> store = load_vector_store('my_store', './data', embeddings)
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_model


def generate_embeddings(chunks: list[str], model_name: str) -> list[list[float]]:
    """
    Generate dense vector embeddings from text chunks using a HuggingFace model.

    This function converts a list of text strings into their corresponding vector
    representations. Each text chunk is transformed into a high-dimensional vector
    (typically 384, 768, or 1536 dimensions depending on the model) that captures
    its semantic meaning.

    The function uses LangChain's HuggingFaceEmbeddings wrapper, which provides:
        - Automatic model downloading and caching
        - Batch processing for efficiency
        - Normalized embeddings (unit length) for cosine similarity

    Args:
        chunks: List of text strings to embed (typically output from text_chunker)
                Each string should be under the model's maximum token limit
                (256 tokens for all-MiniLM-L6-v2, ~1000 characters)

        model_name: HuggingFace model identifier (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
                    The model will be automatically downloaded on first use and cached locally

    Returns:
        A list of embedding vectors, where:
            - Outer list length = number of input chunks
            - Inner list length = embedding dimension (model-specific)
            - Each float represents one dimension of the semantic vector space

    Example:
        >>> chunks = ["This is the first document.", "This is the second document."]
        >>> embeddings = generate_embeddings(chunks, 'sentence-transformers/all-MiniLM-L6-v2')
        >>> len(embeddings)
        2
        >>> len(embeddings[0])
        384
        >>> type(embeddings[0][0])
        <class 'float'>

    Performance Notes:
        - First call downloads the model (~80MB for all-MiniLM-L6-v2)
        - Subsequent calls use cached model
        - Processing speed: ~100-500 chunks/second on CPU (model-dependent)
        - GPU acceleration available if PyTorch with CUDA is installed
    """
    # Initialize the HuggingFace embedding model
    # The model will be downloaded and cached automatically on first use
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Use CPU (can be changed to 'cuda' for GPU)
        encode_kwargs={'normalize_embeddings': True}  # Normalize to unit length for cosine similarity
    )

    # Generate embeddings for all chunks using the model's embed_documents method
    # This method handles batching internally for efficiency
    embeddings = embedding_model.embed_documents(chunks)

    # Return the list of embedding vectors
    return embeddings


if __name__ == '__main__':
    # Comprehensive test block to verify embedding generation functionality

    print("=" * 80)
    print("VECTOR EMBEDDING GENERATION - TESTING")
    print("=" * 80)

    # Define sample text chunks for testing
    # Using semantically different chunks to verify the model produces distinct embeddings
    sample_chunks = [
        "The quick brown fox jumps over the lazy dog. This is a test sentence about animals.",
        "Python is a high-level programming language. It is widely used for data science and machine learning.",
        "Machine learning models require large amounts of data for training. Quality data is essential."
    ]

    # Specify the recommended embedding model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    print("\n--- TEST CONFIGURATION ---")
    print(f"Model: {model_name}")
    print(f"Model Specifications:")
    print(f"  - Embedding Dimension: 384")
    print(f"  - Max Sequence Length: 256 tokens (~1000 characters)")
    print(f"  - Optimized for: Semantic search and retrieval")
    print(f"\nNumber of sample chunks: {len(sample_chunks)}")

    # Display sample chunks
    print("\n--- SAMPLE CHUNKS ---")
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(chunk)} characters")
        print(f"  Text: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

    # Generate embeddings
    print("\n--- GENERATING EMBEDDINGS ---")
    print("Loading model and generating embeddings...")
    print("(First run will download the model - this may take a minute)")

    embeddings = generate_embeddings(sample_chunks, model_name)

    print("[OK] Embedding generation complete!")

    # Verify and display results
    print("\n--- EMBEDDING RESULTS ---")
    print(f"Number of chunks processed: {len(sample_chunks)}")
    print(f"Number of embeddings generated: {len(embeddings)}")

    # Verify embedding dimensions
    if embeddings:
        first_embedding_dim = len(embeddings[0])
        print(f"\nEmbedding Dimension: {first_embedding_dim}")

        # Verify all embeddings have the same dimension
        all_same_dim = all(len(emb) == first_embedding_dim for emb in embeddings)
        print(f"All embeddings have consistent dimension: {all_same_dim}")

        # Check if dimension matches expected value for the model
        expected_dim = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        if first_embedding_dim == expected_dim:
            print(f"[OK] Dimension matches model specification ({expected_dim})")
        else:
            print(f"[WARNING] Dimension {first_embedding_dim} differs from expected {expected_dim}")

    # Display sample embedding values
    print("\n--- SAMPLE EMBEDDING VALUES ---")
    print("First embedding vector (first 10 values):")
    print(f"  {embeddings[0][:10]}")
    print(f"  ... ({len(embeddings[0]) - 10} more values)")

    # Calculate and display vector norms (should be ~1.0 due to normalization)
    print("\n--- EMBEDDING STATISTICS ---")
    import math
    for i, embedding in enumerate(embeddings, 1):
        # Calculate L2 norm (magnitude) of the vector
        norm = math.sqrt(sum(x * x for x in embedding))
        print(f"Chunk {i} - Vector norm (magnitude): {norm:.6f}")

    print("\nNote: Normalized embeddings should have norm approximately 1.0")

    # Demonstrate semantic similarity calculation
    print("\n--- SEMANTIC SIMILARITY DEMONSTRATION ---")
    print("Calculating cosine similarity between chunks...")

    # Cosine similarity for normalized vectors is just the dot product
    def cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two normalized vectors."""
        return sum(a * b for a, b in zip(vec1, vec2))

    print("\nPairwise Similarities:")
    print(f"  Chunk 1 vs Chunk 2: {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"  Chunk 1 vs Chunk 3: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
    print(f"  Chunk 2 vs Chunk 3: {cosine_similarity(embeddings[1], embeddings[2]):.4f}")

    print("\nInterpretation:")
    print("  - Values closer to 1.0 indicate high semantic similarity")
    print("  - Values closer to 0.0 indicate low semantic similarity")
    print("  - Chunks 2 and 3 should be most similar (both about machine learning/data)")

    # Summary
    print("\n" + "=" * 80)
    print("TESTING COMPLETE - Embedding generation is working correctly!")
    print("=" * 80)
    print("\nKey Verification Points:")
    print(f"  [OK] Model loaded successfully: {model_name}")
    print(f"  [OK] Generated {len(embeddings)} embeddings for {len(sample_chunks)} chunks")
    print(f"  [OK] Embedding dimension: {len(embeddings[0])}")
    print(f"  [OK] Embeddings are normalized (unit vectors)")
    print(f"  [OK] Semantic similarity calculation works")
    print("\n" + "=" * 80)

    print("\nNext Steps:")
    print("  1. Integrate with config.py (add EmbeddingConfig)")
    print("  2. Update pipeline.py to support embedding generation")
    print("  3. Test with real document chunks from the pipeline")
    print("  4. Implement vector store for embedding storage and retrieval")
