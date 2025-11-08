"""
Text Chunking Module

This module provides functionality to split long text into smaller, semantically coherent chunks.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent text segmentation that preserves
meaning and context across chunk boundaries.

This is a critical component for preparing text for vector embeddings and retrieval systems,
as it ensures text segments are appropriately sized for embedding models while maintaining
semantic coherence.

Dependencies required:
    pip install langchain langchain-text-splitters
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split long text into smaller, semantically coherent chunks.

    This function uses LangChain's RecursiveCharacterTextSplitter, which intelligently
    splits text by attempting to preserve natural document structure. It tries to split
    on paragraph breaks first, then sentences, then words, and only breaks mid-word as
    a last resort.

    The splitting hierarchy:
        1. Double newlines (paragraph breaks): \\n\\n
        2. Single newlines: \\n
        3. Spaces (word boundaries)
        4. Individual characters (last resort)

    Args:
        text: The input text to be split into chunks (typically clean, normalized text)
        chunk_size: Target maximum size for each chunk in characters (default: 1000)
        chunk_overlap: Number of characters to overlap between consecutive chunks (default: 200)
                      Overlap ensures concepts at chunk boundaries are preserved in at least
                      one complete chunk, improving retrieval quality.

    Returns:
        A list of text chunks (strings), each respecting the chunk_size limit while
        preserving semantic coherence through intelligent splitting.

    Example:
        >>> long_text = "This is a very long document..." * 100
        >>> chunks = chunk_text(long_text, chunk_size=1000, chunk_overlap=200)
        >>> print(f"Created {len(chunks)} chunks")
        Created 15 chunks
    """
    # Initialize the RecursiveCharacterTextSplitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Use character count for measuring chunk size
        separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
    )

    # Split the text and return the list of chunks
    chunks = text_splitter.split_text(text)

    return chunks


if __name__ == '__main__':
    # Test the chunk_text function with a long sample document

    # Sample long text simulating a business document (RFP-style content)
    sample_text = """
    Request for Proposal: Enterprise Document Management System

    1. PROJECT OVERVIEW

    Our organization is seeking proposals from qualified vendors to provide a comprehensive Enterprise Document Management System (EDMS) that will support our document lifecycle management needs across all departments.

    The selected solution must provide robust capabilities for document creation, storage, retrieval, version control, and collaboration. The system should support multiple file formats including PDF, DOCX, XLSX, and image files.

    2. TECHNICAL REQUIREMENTS

    2.1 System Architecture
    The proposed solution must be built on a modern, scalable architecture that supports both cloud-based and on-premises deployment options. The system should utilize microservices architecture to ensure modularity and ease of maintenance.

    High availability is critical. The system must guarantee 99.9% uptime with automatic failover capabilities. Load balancing across multiple servers is required to handle peak usage periods.

    2.2 Security and Compliance
    All data must be encrypted both in transit and at rest using industry-standard encryption protocols (minimum AES-256). The system must support role-based access control (RBAC) with granular permission settings.

    Compliance with GDPR, HIPAA, and SOC 2 Type II standards is mandatory. The vendor must provide documentation of compliance certifications and undergo regular third-party security audits.

    2.3 Document Processing Capabilities
    The system must include intelligent document processing features such as optical character recognition (OCR) for scanned documents, automatic metadata extraction, and full-text search capabilities.

    Advanced search functionality should support Boolean operators, fuzzy matching, and search within specific metadata fields. Search results must be returned within 2 seconds for 95% of queries.

    3. FUNCTIONAL REQUIREMENTS

    3.1 User Interface
    The user interface must be intuitive and accessible via web browsers (Chrome, Firefox, Safari, Edge) without requiring plugins. Mobile applications for iOS and Android are required.

    The interface should support drag-and-drop file uploads, bulk operations, and customizable dashboards. Users must be able to preview documents without downloading them.

    3.2 Collaboration Features
    Real-time collaborative editing capabilities are required for supported document formats. The system must track all changes with a complete audit trail showing who made what changes and when.

    Comment and annotation features should allow users to provide feedback on documents without altering the original content. Notification systems must alert users to relevant document activities.

    3.3 Workflow Automation
    The system must include a visual workflow designer that allows non-technical users to create and modify approval workflows. Support for parallel and sequential approval chains is required.

    Automated routing based on document type, content, or metadata should be supported. The system must integrate with email systems to send notifications and allow email-based approvals.

    4. INTEGRATION REQUIREMENTS

    4.1 Third-Party Integration
    The solution must provide RESTful APIs for integration with existing enterprise systems including ERP, CRM, and HR management systems. API documentation must be comprehensive and include code examples.

    Pre-built connectors for common enterprise applications (Microsoft Office 365, Google Workspace, Salesforce, SAP) are highly desirable. The vendor should maintain and update these connectors regularly.

    4.2 Data Migration
    The vendor must provide tools and services to migrate existing documents from our current systems. Migration must preserve all metadata, version history, and access permissions.

    A detailed migration plan with timeline, testing procedures, and rollback capabilities must be provided. The vendor should estimate the migration timeline based on our data volume of approximately 5 million documents.

    5. PERFORMANCE REQUIREMENTS

    The system must support a minimum of 5,000 concurrent users without performance degradation. Document upload speeds should achieve at least 10 MB/s for users with standard corporate network connections.

    Batch processing capabilities must handle at least 10,000 documents per hour for bulk operations such as metadata updates or format conversions.

    6. VENDOR QUALIFICATIONS

    Vendors must have a minimum of 5 years of experience implementing enterprise document management systems. At least 3 references from organizations of similar size and industry must be provided.

    The vendor must demonstrate financial stability and commitment to ongoing product development. A product roadmap for the next 24 months should be included in the proposal.

    7. SUPPORT AND TRAINING

    Comprehensive training programs for administrators and end-users must be provided. Training should be available in multiple formats including on-site, virtual instructor-led, and self-paced online courses.

    Ongoing support must include 24/7 availability for critical issues with guaranteed response times of 1 hour for severity-1 incidents. A dedicated customer success manager should be assigned to our account.

    8. SUBMISSION REQUIREMENTS

    Proposals must be submitted by 5:00 PM EST on December 15, 2024. Late submissions will not be accepted. All proposals should include detailed pricing, implementation timeline, and references.

    Questions regarding this RFP should be submitted in writing to procurement@example.com by November 30, 2024.
    """

    # Remove leading/trailing whitespace from sample
    sample_text = sample_text.strip()

    # Configuration parameters (using recommended starting values)
    chunk_size = 1000
    chunk_overlap = 200

    print("=" * 80)
    print("TEXT CHUNKING - TESTING")
    print("=" * 80)

    print("\n--- INPUT TEXT INFORMATION ---")
    print(f"Total characters in input text: {len(sample_text)}")
    print(f"Total words (approx): {len(sample_text.split())}")
    print(f"Total lines: {sample_text.count(chr(10)) + 1}")

    print("\n--- CHUNKING PARAMETERS ---")
    print(f"Chunk size: {chunk_size} characters")
    print(f"Chunk overlap: {chunk_overlap} characters")
    print(f"Overlap percentage: {(chunk_overlap/chunk_size)*100:.1f}%")

    # Perform the chunking
    print("\n--- PERFORMING CHUNKING ---")
    chunks = chunk_text(sample_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    print("\n--- CHUNKING RESULTS ---")
    print(f"Total chunks created: {len(chunks)}")

    # Calculate statistics
    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunks else 0
    min_chunk_length = min(chunk_lengths) if chunks else 0
    max_chunk_length = max(chunk_lengths) if chunks else 0

    print(f"\nChunk Statistics:")
    print(f"  • Average chunk length: {avg_chunk_length:.0f} characters")
    print(f"  • Minimum chunk length: {min_chunk_length} characters")
    print(f"  • Maximum chunk length: {max_chunk_length} characters")

    # Display individual chunk information
    print("\n--- INDIVIDUAL CHUNK DETAILS ---")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\nChunk {i + 1}:")
        print(f"  Length: {len(chunk)} characters")
        print(f"  First 100 chars: {chunk[:100]}...")
        if i < len(chunks) - 1:
            print(f"  Last 100 chars: ...{chunk[-100:]}")

    # Demonstrate overlap between chunks
    if len(chunks) >= 2:
        print("\n--- OVERLAP VERIFICATION ---")
        print("Checking overlap between Chunk 1 and Chunk 2:\n")

        # Get the end of chunk 1 and beginning of chunk 2
        chunk1_end = chunks[0][-chunk_overlap:]
        chunk2_start = chunks[1][:chunk_overlap]

        print(f"Last {chunk_overlap} chars of Chunk 1:")
        print(f'"{chunk1_end}"')
        print(f"\nFirst {chunk_overlap} chars of Chunk 2:")
        print(f'"{chunk2_start}"')

        # Check if there's actual overlap (some common content)
        # Note: Exact overlap might not match perfectly due to splitting at natural boundaries
        print(f"\nNote: The splitter prioritizes natural boundaries (paragraphs, sentences),")
        print(f"so overlap may not be exactly {chunk_overlap} characters.")

    # Display full content of first chunk for manual inspection
    print("\n--- FIRST CHUNK (FULL CONTENT) ---")
    print("-" * 80)
    print(chunks[0])
    print("-" * 80)

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
