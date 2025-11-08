"""
Query Processor Module
======================
This module provides functions for querying a vector store using LangChain's
native retriever abstractions. It converts a vector store into a retriever
and executes similarity searches to find relevant document chunks.

Required pip installations:
    pip install langchain-community
    pip install langchain-core
    pip install faiss-cpu (or faiss-gpu)
    pip install sentence-transformers

Author: Claude (Senior Python Developer)
Project: Document Structure Extractor - Query Processor Stage
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


def create_retriever(vector_store, k: int = 5) -> VectorStoreRetriever:
    """
    Convert a vector store into a LangChain VectorStoreRetriever object.

    This function wraps the vector store in LangChain's Retriever interface,
    which provides a standardized way to query for relevant documents. The
    retriever will be configured to use similarity search.

    Args:
        vector_store: A loaded vector store object (e.g., FAISS, Chroma)
        k (int): Number of top documents to retrieve (default: 5)

    Returns:
        VectorStoreRetriever: A LangChain retriever object configured for
                              similarity-based document retrieval

    Example:
        >>> from vector_store_manager import load_vector_store
        >>> store = load_vector_store('./data/vector_store')
        >>> retriever = create_retriever(store, k=3)
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",  # Use similarity search (alternatives: "mmr" for diversity)
        search_kwargs={"k": k}     # Retrieve top-k most similar documents
    )

    return retriever


def query_retriever(retriever: VectorStoreRetriever, query: str) -> List[Document]:
    """
    Execute a retrieval query using the LangChain retriever.

    This function takes a natural language query and uses the retriever to
    find the most relevant document chunks from the vector store. It returns
    standard LangChain Document objects containing both content and metadata.

    Args:
        retriever (VectorStoreRetriever): A LangChain retriever object
        query (str): The user's natural language question or search query

    Returns:
        List[Document]: A list of LangChain Document objects. Each Document has:
                       - page_content (str): The actual text chunk
                       - metadata (dict): Source info (document, page, etc.)

    Example:
        >>> results = query_retriever(retriever, "What are the technical specs?")
        >>> for doc in results:
        ...     print(doc.page_content)
        ...     print(doc.metadata)
    """
    # Use invoke() method (modern LangChain API)
    # Alternative: retriever.get_relevant_documents(query) for older versions
    results = retriever.invoke(query)

    return results


def format_search_results(results: List[Document], show_full_content: bool = False) -> str:
    """
    Format search results into a clean, human-readable string.

    This helper function takes the raw Document objects returned by the
    retriever and formats them for easy reading and verification.

    Args:
        results (List[Document]): List of Document objects from query_retriever
        show_full_content (bool): If True, show full content; if False, show preview

    Returns:
        str: Formatted string representation of the search results

    Example:
        >>> formatted = format_search_results(results)
        >>> print(formatted)
    """
    if not results:
        return "No results found."

    output_lines = []
    output_lines.append(f"\n{'='*70}")
    output_lines.append(f"Found {len(results)} relevant document(s)")
    output_lines.append(f"{'='*70}\n")

    for i, doc in enumerate(results, 1):
        output_lines.append(f"--- Result {i} ---")

        # Display metadata
        if doc.metadata:
            output_lines.append("Metadata:")
            for key, value in doc.metadata.items():
                output_lines.append(f"  {key}: {value}")

        # Display content
        output_lines.append("\nContent:")
        if show_full_content:
            output_lines.append(doc.page_content)
        else:
            # Show preview (first 300 characters)
            preview = doc.page_content[:300]
            if len(doc.page_content) > 300:
                preview += "..."
            output_lines.append(preview)

        output_lines.append("")  # Blank line between results

    return "\n".join(output_lines)


# ============================================================================
# COMPREHENSIVE TEST BLOCK
# ============================================================================

if __name__ == '__main__':
    """
    Test suite for the Query Processor module.

    This block demonstrates the complete workflow:
    1. Load an existing vector store from disk
    2. Create a retriever from the vector store
    3. Execute test queries
    4. Display and verify results

    Prerequisites:
        - A vector store must exist at './data/vector_store'
        - The store should contain embedded document chunks
    """

    import sys

    print("="*70)
    print("Query Processor Module - Comprehensive Test")
    print("="*70)

    # -----------------------------------------------------------------------
    # Step 1: Load the Vector Store
    # -----------------------------------------------------------------------
    print("\n[Step 1] Loading vector store from disk...")

    try:
        from vector_store_manager import load_vector_store
        from embedding_generator import create_embeddings

        # Load the pre-existing vector store
        # Note: The load_vector_store function needs the embedding model
        # to reconstruct the vector store properly
        embeddings = create_embeddings()
        vector_store = load_vector_store(
            store_name='test_vector_store',
            path='./data',
            embedding_model=embeddings
        )

        print("[OK] Vector store loaded successfully")

    except FileNotFoundError:
        print("[ERROR] Vector store not found at './data/test_vector_store'")
        print("  Please ensure you have run the pipeline to create a vector store first.")
        sys.exit(1)
    except ImportError as e:
        print(f"[ERROR] Could not import required modules: {e}")
        print("  Please ensure vector_store_manager.py and embedding_generator.py exist.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error loading vector store: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Create the Retriever
    # -----------------------------------------------------------------------
    print("\n[Step 2] Creating LangChain retriever...")

    try:
        retriever = create_retriever(vector_store, k=3)
        print(f"[OK] Retriever created (configured to retrieve top-3 results)")
        print(f"  Search type: similarity")

    except Exception as e:
        print(f"[ERROR] Error creating retriever: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Define Test Queries
    # -----------------------------------------------------------------------
    print("\n[Step 3] Preparing test queries...")

    # Define a set of test queries that should be relevant to typical
    # business documents (RFPs, specs, catalogs)
    test_queries = [
        "What are the technical specifications?",
        "What are the delivery timelines and schedules?",
        "What compliance requirements are mentioned?",
    ]

    print(f"[OK] Prepared {len(test_queries)} test queries")

    # -----------------------------------------------------------------------
    # Step 4: Execute Queries and Display Results
    # -----------------------------------------------------------------------
    print("\n[Step 4] Executing queries and retrieving results...")
    print("="*70)

    for query_num, query in enumerate(test_queries, 1):
        print(f"\n{'#'*70}")
        print(f"# TEST QUERY {query_num}")
        print(f"{'#'*70}")
        print(f"\nQuery: \"{query}\"")

        try:
            # Execute the retrieval
            results = query_retriever(retriever, query)

            # Display results in a formatted way
            if results:
                print(f"\nRetrieved {len(results)} document(s):\n")

                for i, doc in enumerate(results, 1):
                    print(f"{'-'*70}")
                    print(f"Result {i}:")
                    print(f"{'-'*70}")

                    # Display metadata
                    if doc.metadata:
                        print("\nMetadata:")
                        for key, value in doc.metadata.items():
                            print(f"  • {key}: {value}")

                    # Display content preview
                    print("\nContent Preview:")
                    content_preview = doc.page_content[:250].strip()
                    if len(doc.page_content) > 250:
                        content_preview += "..."
                    print(f"  {content_preview}")
                    print()

            else:
                print("  [WARNING] No results found for this query")

        except Exception as e:
            print(f"  [ERROR] Error executing query: {e}")
            continue

    # -----------------------------------------------------------------------
    # Step 5: Test the Format Helper Function
    # -----------------------------------------------------------------------
    print(f"\n{'#'*70}")
    print("# TESTING FORMAT HELPER FUNCTION")
    print(f"{'#'*70}")

    print("\nExecuting one more query to test format_search_results()...")
    test_format_query = "What information is available about products?"

    try:
        results = query_retriever(retriever, test_format_query)
        formatted_output = format_search_results(results, show_full_content=False)
        print(formatted_output)

    except Exception as e:
        print(f"[ERROR] Error testing format function: {e}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("TEST SUITE COMPLETED")
    print("="*70)
    print("\nManual Verification Checklist:")
    print("  [ ] Do the retrieved documents relate to the query?")
    print("  [ ] Are the top results more relevant than lower-ranked ones?")
    print("  [ ] Is metadata (source, page) correctly preserved?")
    print("  [ ] Is the content preview readable and informative?")
    print("\nIf all checks pass, the Query Processor is ready for integration!")
    print("="*70)
