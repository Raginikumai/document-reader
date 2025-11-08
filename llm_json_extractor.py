"""
LLM JSON Extractor Module
==========================
This module provides functionality for extracting structured JSON from retrieved
document chunks using Groq LLM. It implements a single-pass extraction strategy
with careful prompt engineering to ensure accurate, factual extraction.

The core function is stateless and composable, designed to be reusable in more
advanced workflows (e.g., iterative extraction, two-pass strategies).

Required pip installations:
    pip install langchain-groq
    pip install langchain-core
    pip install python-dotenv

Author: Claude (Senior Python Developer)
Project: Document Structure Extractor - LLM JSON Extraction Stage
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

import sys
from pathlib import Path as PathLib
# Add modules directory to Python path so imports work after refactoring
sys.path.insert(0, str(PathLib(__file__).parent / 'modules'))


def get_default_json_schema() -> Dict[str, Any]:
    """
    Return the default JSON schema structure for document extraction.

    This is the final "Middle Ground" schema - perfectly balanced between
    simplicity and robustness. It extracts items with their specifications,
    support/compliance details, and quantity information without ambiguity.

    Returns:
        Dictionary defining the expected JSON structure with descriptions

    Schema Design Principles:
        - Item-focused: Only extract items (products, services, equipment)
        - Clear boundaries: Technical specs vs support/compliance clearly separated
        - Warranty clarity: Warranty has designated place in support_and_compliance
        - Quantity tracking: Captures procurement quantities when present
        - No ambiguity: Each piece of information has one clear home
    """
    schema = {
        "items": [
            {
                "item_name": "The name of the product, service, or item",
                "key_info": {
                    "category": "A relevant category for the item",
                    "description": "A concise summary or description of the item.",
                    "quantity": "The number of units required for this item (if mentioned)."
                },
                "technical_specifications": {
                    "spec_name_1": "spec_value_1"
                },
                "support_and_compliance": {
                    "warranty": "The warranty period and terms for this specific item.",
                    "certifications": "Any required certifications for this item (e.g., BIS, CE).",
                    "vendor_requirements": "Any vendor qualifications directly related to providing this item."
                }
            }
        ]
    }

    return schema


def format_documents_as_context(retrieved_docs: List[Document]) -> str:
    """
    Format a list of retrieved Document objects into a clear context string.

    This function creates a well-structured presentation of document chunks
    that the LLM can easily parse and reference.

    Args:
        retrieved_docs: List of LangChain Document objects from retrieval

    Returns:
        Formatted string with numbered chunks and metadata

    Format:
        --- Document Chunk 1 ---
        Source: filename.txt
        Chunk ID: 0
        Content:
        [chunk text here]

        [repeated for each document]
    """
    if not retrieved_docs:
        return "No context documents provided."

    context_parts = []

    for i, doc in enumerate(retrieved_docs, 1):
        chunk_section = f"--- Document Chunk {i} ---\n"

        # Add metadata
        if doc.metadata:
            source = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 'Unknown')
            chunk_section += f"Source: {source}\n"
            chunk_section += f"Chunk ID: {chunk_id}\n"

        # Add content
        chunk_section += f"Content:\n{doc.page_content}\n"

        context_parts.append(chunk_section)

    return "\n".join(context_parts)


def extract_json_from_context(
    query: str,
    retrieved_docs: List[Document],
    llm: ChatGroq,
    json_parser: JsonOutputParser,
    custom_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract structured JSON from retrieved document context using Groq LLM.

    This is the core, stateless extraction function. It takes all dependencies
    as explicit arguments, making it composable and reusable in various workflows.

    Args:
        query: The original user query/question
        retrieved_docs: List of LangChain Document objects from vector search
        llm: Initialized ChatGroq LLM instance
        json_parser: JsonOutputParser instance for parsing LLM response
        custom_schema: Optional custom JSON schema (uses default if None)

    Returns:
        Parsed JSON as a Python dictionary

    Raises:
        ValueError: If retrieved_docs is empty
        Exception: If LLM call or JSON parsing fails

    Example:
        >>> llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        >>> parser = JsonOutputParser()
        >>> result = extract_json_from_context(query, docs, llm, parser)
        >>> print(result['extracted_information']['summary'])

    Design Notes:
        - Stateless: No hidden state, all inputs explicit
        - Composable: Can be wrapped in iteration, caching, etc.
        - Testable: Easy to mock dependencies
        - Reusable: Same function for single-pass or multi-pass workflows
    """
    # Validation
    if not retrieved_docs:
        raise ValueError("No documents provided for extraction. retrieved_docs cannot be empty.")

    # Use custom schema or default
    schema = custom_schema if custom_schema is not None else get_default_json_schema()

    # Format retrieved documents into context string
    context = format_documents_as_context(retrieved_docs)

    # Get current timestamp for metadata
    timestamp = datetime.now().isoformat()

    # Construct the prompt using LangChain's ChatPromptTemplate
    # This uses a system message (role definition) and user message (task)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert document analyzer specializing in extracting product and service information from business documents like RFPs, technical specifications, and product catalogs.

Your task is to intelligently identify and extract ONLY items (products, services, equipment) with their complete information from the provided context.

=== DEFINITION OF AN "ITEM" ===
An "item" is a tangible product, piece of equipment, service, or component that is being procured, described, or specified. Items can be identified in two ways:

1. EXPLICIT ITEMS: Clearly labeled with keywords like "ITEM:", "Product:", "Equipment:", or in structured lists
   Example: "ITEM: Network Switch" or "Product Name: Business Laptop"

2. IMPLICIT ITEMS: The central subject of a sentence or paragraph with associated properties/specifications
   Examples:
   - "The project requires a new user authentication module with support for MFA" → Item: "User authentication module"
   - "We need managed switches capable of handling 10Gbps throughput" → Item: "Managed switches"
   - "Supply and install backup power systems with 6kVA capacity" → Item: "Backup power systems"

Key principle: If something has specifications, quantities, or requirements attached to it, it's likely an item.

=== WHAT IS NOT AN ITEM ===
- Vendor qualifications (e.g., "Vendor must have 3 years experience")
- Evaluation criteria (e.g., "Technical compliance: 70 points")
- General delivery requirements (e.g., "Delivery within 30 days")
- Document metadata (e.g., "RFP Number: XYZ-123")

CRITICAL INSTRUCTIONS:
1. Use intelligent detection to find ALL items, both explicit and implicit
2. Be PRECISE and FACTUAL - only extract information explicitly stated in the context
3. Do NOT hallucinate or invent specifications not present
4. Extract quantity whenever mentioned (e.g., "Quantity: 75" or "Supply 10 units")
5. Place warranty/support information in support_and_compliance.warranty
6. Place certifications (BIS, CE, ISO, etc.) in support_and_compliance.certifications
7. Use technical_specifications for objective technical details only
8. Your response must be VALID JSON ONLY - no markdown, no code blocks, no explanations
9. Start directly with opening curly brace, end with closing curly brace"""),

        ("human", """QUERY:
{query}

CONTEXT:
{context}

TASK:
Extract ALL items (products, services, equipment) from the context above using this exact JSON schema:

{schema}

FIELD-SPECIFIC EXTRACTION RULES:

1. item_name:
   - Extract the clear, specific name of the item
   - Use the exact term from the document when possible
   - For implicit items, use the noun phrase that describes it

2. key_info.category:
   - Classify the item (e.g., "Networking Hardware", "Computer Equipment", "Power Systems", "Software Module")
   - Use industry-standard categories when possible

3. key_info.description:
   - Provide a concise 1-2 sentence description
   - Focus on what the item IS and its primary purpose

4. key_info.quantity:
   - Extract if mentioned (e.g., "Quantity: 75", "Supply 10 units", "2 pieces")
   - If not mentioned, use: "Not specified"

5. technical_specifications:
   - Include ONLY objective, measurable technical details
   - Examples: Processor type, RAM size, Port count, Dimensions, Power rating, Capacity
   - Use clear key names: "Processor", "Memory (RAM)", "Port Count", "Storage Capacity"
   - Format: {{"spec_name": "spec_value"}}
   - If no technical specs found, use empty object: {{}}

6. support_and_compliance.warranty:
   - Extract warranty period and type (e.g., "3-year comprehensive on-site support")
   - If not mentioned, use: "Not specified"

7. support_and_compliance.certifications:
   - Extract any required certifications, standards, or compliance requirements
   - Examples: "BIS certified", "CE marked", "ISO 9001 compliant"
   - If not mentioned, use: "Not specified"

8. support_and_compliance.vendor_requirements:
   - Extract ONLY item-specific vendor qualifications
   - Example: "Vendor must be authorized service provider for this product"
   - If not mentioned, use: "Not specified"

IMPORTANT: If no items are found, return: {{"items": []}}

OUTPUT:
Return ONLY valid JSON matching the schema exactly. Begin now:""")
    ])

    # Create the LangChain Expression Language (LCEL) chain
    # This composes: prompt template → LLM → JSON parser
    extraction_chain = prompt_template | llm | json_parser

    # Execute the chain
    result = extraction_chain.invoke({
        "query": query,
        "context": context,
        "schema": json.dumps(schema, indent=2)
    })

    # The simplified schema doesn't include metadata
    # Just return the result as-is (should contain only "items" array)
    return result


def validate_extraction_quality(extraction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive validation on the extraction result.

    This function checks for common issues and returns a detailed validation report
    tailored to the "Middle Ground" schema structure.

    Args:
        extraction_result: The dictionary returned by extract_json_from_context

    Returns:
        Dictionary with validation results:
            {
                "is_valid": bool,
                "issues": List[str],
                "warnings": List[str],
                "statistics": Dict[str, int]
            }
    """
    issues = []
    warnings = []
    statistics = {}

    # Check for required top-level key
    if "items" not in extraction_result:
        issues.append("Missing required key: 'items'")
        return {
            "is_valid": False,
            "issues": issues,
            "warnings": warnings,
            "statistics": statistics
        }

    # Gather statistics
    items = extraction_result.get("items", [])
    statistics["items_count"] = len(items)

    # Check if extraction is empty
    if statistics["items_count"] == 0:
        warnings.append("No items were extracted from the context")

    # Validate each item has required fields and proper structure
    items_with_specs = 0
    items_with_quantity = 0
    items_with_warranty = 0
    items_with_certifications = 0

    for i, item in enumerate(items, 1):
        # Validate item_name
        if "item_name" not in item:
            issues.append(f"Item {i} missing 'item_name'")

        # Validate key_info
        if "key_info" not in item:
            issues.append(f"Item {i} missing 'key_info'")
        elif not isinstance(item.get("key_info"), dict):
            issues.append(f"Item {i} 'key_info' must be a dictionary")
        else:
            key_info = item["key_info"]
            # Check key_info required fields
            if "category" not in key_info:
                warnings.append(f"Item {i} missing 'category' in key_info")
            if "description" not in key_info:
                warnings.append(f"Item {i} missing 'description' in key_info")
            if "quantity" in key_info and key_info["quantity"] != "Not specified":
                items_with_quantity += 1

        # Validate technical_specifications
        if "technical_specifications" not in item:
            warnings.append(f"Item {i} missing 'technical_specifications'")
        elif not isinstance(item.get("technical_specifications"), dict):
            issues.append(f"Item {i} 'technical_specifications' must be a dictionary")
        else:
            if item["technical_specifications"]:  # Has specs
                items_with_specs += 1

        # Validate support_and_compliance
        if "support_and_compliance" not in item:
            warnings.append(f"Item {i} missing 'support_and_compliance'")
        elif not isinstance(item.get("support_and_compliance"), dict):
            issues.append(f"Item {i} 'support_and_compliance' must be a dictionary")
        else:
            support = item["support_and_compliance"]
            # Check support_and_compliance fields
            if "warranty" in support and support["warranty"] != "Not specified":
                items_with_warranty += 1
            if "certifications" in support and support["certifications"] != "Not specified":
                items_with_certifications += 1

    # Compile statistics
    statistics["items_with_specifications"] = items_with_specs
    statistics["items_without_specifications"] = statistics["items_count"] - items_with_specs
    statistics["items_with_quantity"] = items_with_quantity
    statistics["items_with_warranty"] = items_with_warranty
    statistics["items_with_certifications"] = items_with_certifications

    validation_result = {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "statistics": statistics
    }

    return validation_result


# ============================================================================
# COMPREHENSIVE TEST BLOCK
# ============================================================================

if __name__ == '__main__':
    """
    Comprehensive test suite for the LLM JSON Extractor module.

    This demonstrates the complete end-to-end workflow:
    1. Load vector store
    2. Perform query and retrieval
    3. Initialize LLM and parser
    4. Extract JSON from context
    5. Validate and display results

    Prerequisites:
        - Vector store exists at './data/test_vector_store'
        - GROQ_API_KEY environment variable is set
    """

    import sys
    from dotenv import load_dotenv
    from config import PipelineConfig

    # Load environment variables from .env file
    load_dotenv()

    print("=" * 80)
    print("LLM JSON EXTRACTOR - COMPREHENSIVE TEST")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 1: Verify API Key
    # -----------------------------------------------------------------------
    print("\n[Step 1] Verifying Groq API key...")

    api_key = PipelineConfig.llm.GROQ_API_KEY
    if not api_key:
        print("[ERROR] GROQ_API_KEY environment variable not set!")
        print("\nPlease set your API key:")
        print("  - Create a .env file with: GROQ_API_KEY=your_key_here")
        print("  - Or set environment variable: export GROQ_API_KEY=your_key_here")
        print("  - Get your API key from: https://console.groq.com/keys")
        sys.exit(1)

    print(f"[OK] API key found (length: {len(api_key)} characters)")

    # -----------------------------------------------------------------------
    # Step 2: Load Vector Store
    # -----------------------------------------------------------------------
    print("\n[Step 2] Loading vector store...")

    try:
        from vector_store_manager import load_vector_store
        from embedding_generator import create_embeddings

        embeddings = create_embeddings()
        vector_store = load_vector_store(
            store_name='test_vector_store',
            path='./data/vector_stores',
            embedding_model=embeddings
        )

        print("[OK] Vector store loaded successfully")

    except FileNotFoundError:
        print("[ERROR] Vector store not found at './data/vector_stores/test_vector_store'")
        print("  Please run pipeline.py first to create the test vector store")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load vector store: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 3: Perform Query and Retrieval
    # -----------------------------------------------------------------------
    print("\n[Step 3] Performing query and retrieval...")

    try:
        from query_processor import create_retriever, query_retriever

        # Create retriever with k=5 for more context
        retriever = create_retriever(vector_store, k=5)

        # Test query
        test_query = "What are the technical specifications for the networking equipment?"

        print(f"\nQuery: \"{test_query}\"")

        # Retrieve relevant documents
        retrieved_docs = query_retriever(retriever, test_query)

        print(f"[OK] Retrieved {len(retrieved_docs)} document chunks")

        # Display retrieved chunks summary
        print("\nRetrieved chunks:")
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 'Unknown')
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"  {i}. Source: {source}, Chunk: {chunk_id}")
            print(f"     Preview: {preview}...")

    except Exception as e:
        print(f"[ERROR] Query/retrieval failed: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 4: Initialize LLM and Parser
    # -----------------------------------------------------------------------
    print("\n[Step 4] Initializing Groq LLM and JSON parser...")

    try:
        # Initialize Groq LLM with configuration from PipelineConfig
        # All settings are pulled from the centralized configuration
        llm = ChatGroq(
            model=PipelineConfig.llm.MODEL_NAME,
            temperature=PipelineConfig.llm.TEMPERATURE,
            max_tokens=PipelineConfig.llm.MAX_TOKENS,
            groq_api_key=api_key,
            timeout=PipelineConfig.llm.TIMEOUT
        )

        # Initialize JSON output parser
        json_parser = JsonOutputParser()

        print(f"[OK] LLM initialized: {PipelineConfig.llm.MODEL_NAME}")
        print("[OK] JSON parser initialized")
        print("\nLLM Configuration (from PipelineConfig):")
        print(f"  Provider: Groq")
        print(f"  Model: {PipelineConfig.llm.MODEL_NAME}")
        print(f"  Temperature: {PipelineConfig.llm.TEMPERATURE} (deterministic)")
        print(f"  Max tokens: {PipelineConfig.llm.MAX_TOKENS}")
        print(f"  Timeout: {PipelineConfig.llm.TIMEOUT} seconds")

    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 5: Extract JSON from Context
    # -----------------------------------------------------------------------
    print("\n[Step 5] Extracting structured JSON from retrieved context...")
    print("\n" + "=" * 80)
    print("CALLING GROQ LLM FOR EXTRACTION...")
    print("=" * 80)

    try:
        # Call the core extraction function
        extraction_result = extract_json_from_context(
            query=test_query,
            retrieved_docs=retrieved_docs,
            llm=llm,
            json_parser=json_parser
        )

        print("\n[OK] JSON extraction completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 6: Validate Extraction Quality
    # -----------------------------------------------------------------------
    print("\n[Step 6] Validating extraction quality...")

    validation = validate_extraction_quality(extraction_result)

    print(f"\nValidation Results:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Issues: {len(validation['issues'])}")
    print(f"  Warnings: {len(validation['warnings'])}")

    if validation['issues']:
        print("\n  Issues found:")
        for issue in validation['issues']:
            print(f"    - {issue}")

    if validation['warnings']:
        print("\n  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")

    print("\n  Extraction Statistics:")
    for key, value in validation['statistics'].items():
        print(f"    - {key}: {value}")

    # -----------------------------------------------------------------------
    # Step 7: Display Results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXTRACTION RESULT (PRETTY-PRINTED JSON)")
    print("=" * 80)

    # Pretty-print the JSON result
    print("\n" + json.dumps(extraction_result, indent=2, ensure_ascii=False))

    # -----------------------------------------------------------------------
    # Step 8: Display Key Findings Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXTRACTED ITEMS SUMMARY (Middle Ground Schema)")
    print("=" * 80)

    # Extract and display items
    if "items" in extraction_result:
        items = extraction_result["items"]

        if items:
            print(f"\nTotal Items Extracted: {len(items)}")
            print("\nItem Details:")
            for i, item in enumerate(items, 1):
                # Basic item information
                name = item.get("item_name", "Unknown")
                key_info = item.get("key_info", {})
                category = key_info.get("category", "Unknown")
                description = key_info.get("description", "No description")
                quantity = key_info.get("quantity", "Not specified")

                # Technical specifications
                specs = item.get("technical_specifications", {})

                # Support and compliance
                support = item.get("support_and_compliance", {})
                warranty = support.get("warranty", "Not specified")
                certifications = support.get("certifications", "Not specified")
                vendor_req = support.get("vendor_requirements", "Not specified")

                # Display item
                print(f"\n  {'='*70}")
                print(f"  ITEM {i}: {name}")
                print(f"  {'='*70}")
                print(f"  Category: {category}")
                print(f"  Quantity: {quantity}")
                print(f"  Description: {description}")

                # Technical specifications
                if specs:
                    print(f"\n  Technical Specifications ({len(specs)} total):")
                    for spec_name, spec_value in specs.items():
                        print(f"    • {spec_name}: {spec_value}")
                else:
                    print(f"\n  Technical Specifications: None found")

                # Support and compliance
                print(f"\n  Support & Compliance:")
                print(f"    • Warranty: {warranty}")
                print(f"    • Certifications: {certifications}")
                if vendor_req != "Not specified":
                    print(f"    • Vendor Requirements: {vendor_req}")

        else:
            print("\nNo items were extracted from the context.")

    # -----------------------------------------------------------------------
    # Step 9: Manual Verification Checklist
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MANUAL VERIFICATION CHECKLIST (Middle Ground Schema)")
    print("=" * 80)

    print("\nPlease verify the following:")
    print("  [ ] Are all item names accurate and complete?")
    print("  [ ] Do categories make sense for each item?")
    print("  [ ] Are descriptions concise and factual?")
    print("  [ ] Are quantities correctly extracted (when present)?")
    print("  [ ] Are technical specifications factual (no hallucinations)?")
    print("  [ ] Is warranty information placed in support_and_compliance.warranty?")
    print("  [ ] Are certifications placed in support_and_compliance.certifications?")
    print("  [ ] Is the JSON structure following the Middle Ground schema exactly?")
    print("  [ ] Were both explicit AND implicit items detected?")

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nNext Steps:")
    print("  1. Review extracted items for accuracy and completeness")
    print("  2. Verify intelligent detection found both explicit and implicit items")
    print("  3. Confirm warranty is in support_and_compliance (not tech specs)")
    print("  4. Check that quantities were extracted when present")
    print("  5. Validate certifications are properly captured")
    print("  6. Test with different document types (catalogs, specs, RFPs)")
    print("  7. Build your knowledge base using this Middle Ground schema")

    print("\n" + "=" * 80)
    print("Middle Ground LLM JSON Extractor is ready for production use!")
    print("=" * 80)
