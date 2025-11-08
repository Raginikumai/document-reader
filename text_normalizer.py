"""
Text Normalization Module

This module provides functionality to clean and normalize raw text extracted from documents.
It removes excessive whitespace, standardizes line breaks, and prepares text for downstream
processing (chunking, embeddings, etc.).

Dependencies:
    re (standard library - no installation required)
"""

import re


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
    # This regex matches 2 or more spaces and replaces them with a single space
    text = re.sub(r' {2,}', ' ', text)

    # Step 4: Collapse multiple consecutive newlines into a maximum of two
    # This preserves paragraph breaks (double newlines) but removes excessive blank lines
    # Matches 3 or more newlines and replaces them with exactly 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Step 5: Strip leading and trailing whitespace from the entire text
    text = text.strip()

    return text


if __name__ == '__main__':
    # Test the normalize_text function with a messy sample string

    # Create a sample string with various whitespace issues
    messy_text = """
    This    text   has     too     many    spaces.




    It also has way too many blank lines between paragraphs.

\tThis line starts with a tab character.
Here are some tabs in the middle:\t\t\tof the line.

Mixed line breaks follow:\r\nWindows style line break here.\rOld Mac style here.

    Leading and trailing spaces everywhere.

Multiple              spaces           scattered          throughout.


Final paragraph with clean text to preserve.
    """

    print("=" * 70)
    print("TEXT NORMALIZATION - TESTING")
    print("=" * 70)

    print("\n--- ORIGINAL MESSY TEXT ---")
    print("-" * 70)
    print(repr(messy_text))  # Using repr() to show invisible characters
    print("-" * 70)
    print(f"Original length: {len(messy_text)} characters")

    # Normalize the text
    cleaned_text = normalize_text(messy_text)

    print("\n--- NORMALIZED CLEAN TEXT ---")
    print("-" * 70)
    print(repr(cleaned_text))  # Using repr() to show the cleaned result
    print("-" * 70)
    print(f"Cleaned length: {len(cleaned_text)} characters")

    print("\n--- VISUAL COMPARISON ---")
    print("-" * 70)
    print("CLEANED TEXT (normal display):")
    print(cleaned_text)
    print("-" * 70)

    # Character count comparison
    reduction = len(messy_text) - len(cleaned_text)
    reduction_percent = (reduction / len(messy_text)) * 100

    print("\n--- STATISTICS ---")
    print(f"Characters removed: {reduction}")
    print(f"Reduction: {reduction_percent:.1f}%")
    print("=" * 70)
