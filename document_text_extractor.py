"""
Document Text Extraction Module - Final Version

This module provides functionality to extract and normalize text from various document formats.
Supports: .txt, .pdf, and .docx files

The module performs two key operations:
    1. Extracts raw text from documents using LangChain loaders
    2. Normalizes the text by removing excessive whitespace and standardizing formatting

Dependencies required:
    pip install langchain langchain-community pypdf docx2txt
    re (standard library - no installation required)

Future enhancements:
    - Advanced error handling
    - Support for additional file formats
"""

import os
import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader


def normalize_text(text: str) -> str:
    """
    Normalize and clean raw text by removing excessive whitespace and standardizing formatting.

    This function performs the following operations:
        1. Replaces tab characters with single spaces
        2. Standardizes line breaks (converts \\r\\n and \\r to \\n)
        3. Collapses multiple consecutive spaces into a single space
        4. Collapses multiple consecutive newlines into a maximum of two (preserves paragraphs)
        5. Strips leading and trailing whitespace from the entire text

    Args:
        text: The raw text string to normalize (may contain irregular whitespace)

    Returns:
        A cleaned, normalized text string with consistent formatting

    Example:
        >>> messy = "Too many    spaces.\\n\\n\\n\\nToo many lines.\\t\\tTabs here."
        >>> clean = normalize_text(messy)
        >>> print(clean)
        Too many spaces.

        Too many lines. Tabs here.
    """
    # Step 1: Replace tabs with single spaces
    text = text.replace('\t', ' ')

    # Step 2: Standardize line breaks (convert \r\n and \r to \n)
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')

    # Step 3: Collapse multiple consecutive spaces into a single space
    text = re.sub(r' {2,}', ' ', text)

    # Step 4: Collapse multiple consecutive newlines into a maximum of two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Step 5: Strip leading and trailing whitespace from the entire text
    text = text.strip()

    return text


def extract_text_from_document(file_path: str) -> str:
    """
    Extract and normalize text content from a document file.

    This function performs two main operations:
        1. Extracts raw text from the document using the appropriate LangChain loader
        2. Normalizes the text to remove excessive whitespace and standardize formatting

    Supported file formats:
        - .txt files (UTF-8 encoding) via TextLoader
        - .pdf files via PyPDFLoader
        - .docx files via Docx2txtLoader

    Args:
        file_path: Path to the document file (absolute or relative)

    Returns:
        A cleaned, normalized string containing the extracted text content

    Raises:
        Returns an error message string for unsupported file types
        (proper exception handling will be added in future versions)

    Example:
        >>> text = extract_text_from_document("sample.pdf")
        >>> print(text[:100])
        This is clean, normalized text from the PDF...
    """
    # Convert to Path object for easier extension handling
    file_path_obj = Path(file_path)

    # Extract file extension (case-insensitive)
    file_extension = file_path_obj.suffix.lower()

    # Initialize loader variable
    loader = None

    # Route to appropriate loader based on file type
    if file_extension == '.txt':
        # Use LangChain's TextLoader with UTF-8 encoding
        loader = TextLoader(file_path, encoding='utf-8')

    elif file_extension == '.pdf':
        # Use LangChain's PyPDFLoader for PDF files
        loader = PyPDFLoader(file_path)

    elif file_extension == '.docx':
        # Use LangChain's Docx2txtLoader for DOCX files
        loader = Docx2txtLoader(file_path)

    else:
        # Return error message for unsupported file types
        # Will be replaced with proper exception handling in future parts
        return f"File type '{file_extension}' is not yet supported. Supported formats: .txt, .pdf, .docx"

    # Load the document (returns a list of Document objects)
    documents = loader.load()

    # Extract and concatenate text from all document objects
    raw_text = ""
    for doc in documents:
        raw_text += doc.page_content

    # Normalize the extracted text before returning
    cleaned_text = normalize_text(raw_text)

    return cleaned_text


if __name__ == '__main__':
    # Test the document extraction and normalization pipeline
    # Replace these paths with actual test file paths on your system

    # Example 1: Test with a .txt file
    test_file_path = "sample_document.txt"

    # Example 2: Test with a .pdf file (uncomment to use)
    # test_file_path = "sample_document.pdf"

    # Example 3: Test with a .docx file (uncomment to use)
    # test_file_path = "sample_document.docx"

    print("=" * 70)
    print("DOCUMENT TEXT EXTRACTION & NORMALIZATION - Testing")
    print("=" * 70)
    print(f"\nFile: {test_file_path}")
    print(f"Format: {Path(test_file_path).suffix.upper()}")
    print("-" * 70)

    # Extract and normalize the text
    result = extract_text_from_document(test_file_path)

    print("\nExtracted & Normalized Text:")
    print("-" * 70)
    # Display first 500 characters to avoid overwhelming output
    if len(result) > 500:
        print(result[:500])
        print("\n... (truncated for display)")
    else:
        print(result)
    print("-" * 70)

    # Statistics
    print("\nStatistics:")
    print(f"  • Total characters: {len(result)}")
    print(f"  • Total words (approx): {len(result.split())}")
    print(f"  • Total lines: {result.count(chr(10)) + 1}")
    print("=" * 70)
    print("\nNote: Text has been cleaned and normalized (whitespace standardized)")
    print("=" * 70)
