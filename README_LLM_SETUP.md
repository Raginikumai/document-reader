# LLM JSON Extractor Setup Guide

## Overview

The LLM JSON Extractor module uses Claude (Anthropic) to extract structured JSON from retrieved document chunks. This guide explains how to set up and use this module.

## Prerequisites

- Python 3.9 or higher
- Anthropic API key (get one at https://console.anthropic.com/)

## Installation

1. **Install required packages:**

```bash
pip install langchain-anthropic langchain-core python-dotenv
```

2. **Set up your API key:**

Create a `.env` file in the project root directory:

```bash
# Copy the example file
cp .env.example .env
```

Then edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=your_actual_api_key_here
```

**Important:** Never commit your `.env` file to version control. It's already in `.gitignore`.

## Usage

### Basic Usage

```python
from llm_json_extractor import extract_json_from_context
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser

# Initialize LLM and parser
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0,
    max_tokens=4096
)
parser = JsonOutputParser()

# Extract JSON from retrieved documents
result = extract_json_from_context(
    query="What are the technical specifications?",
    retrieved_docs=documents,  # From query_processor
    llm=llm,
    json_parser=parser
)

# Access extracted information
print(result['extracted_information']['summary'])
for item in result['extracted_information']['items']:
    print(f"- {item['item_name']}: {item['description']}")
```

### Complete End-to-End Pipeline

```python
# 1. Load vector store
from vector_store_manager import load_vector_store
from embedding_generator import create_embeddings

embeddings = create_embeddings()
vector_store = load_vector_store('test_vector_store', './data', embeddings)

# 2. Query and retrieve
from query_processor import create_retriever, query_retriever

retriever = create_retriever(vector_store, k=5)
docs = query_retriever(retriever, "What are the requirements?")

# 3. Extract JSON
from llm_json_extractor import extract_json_from_context
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
parser = JsonOutputParser()

result = extract_json_from_context(
    query="What are the requirements?",
    retrieved_docs=docs,
    llm=llm,
    json_parser=parser
)
```

## Testing

Run the comprehensive test suite:

```bash
python llm_json_extractor.py
```

This will:
1. Load the test vector store
2. Perform a sample query
3. Extract JSON using Claude
4. Display and validate the results

## JSON Schema

The default schema extracts:

- **document_type**: Type of document (RFP, Specification, etc.)
- **extraction_metadata**: Query, timestamp, confidence
- **extracted_information**:
  - **summary**: Brief overview of findings
  - **items**: Equipment/products with specifications
  - **requirements**: Technical and compliance requirements
  - **timelines**: Deadlines and durations
  - **key_entities**: Organizations, standards, etc.
- **source_documents**: List of source files
- **notes**: Important caveats or observations

Each extracted item includes a `source_chunk` ID for traceability.

## Configuration

LLM settings are managed in `config.py`:

```python
from config import PipelineConfig

# Access LLM configuration
model_name = PipelineConfig.llm.MODEL_NAME  # claude-3-5-sonnet-20241022
temperature = PipelineConfig.llm.TEMPERATURE  # 0.0 (deterministic)
max_tokens = PipelineConfig.llm.MAX_TOKENS  # 4096
```

## Cost Estimation

Typical costs for Claude Sonnet (as of 2024):
- **Per extraction**: $0.02 - $0.08
- **Input tokens**: ~$3 per million tokens
- **Output tokens**: ~$15 per million tokens
- **Cached input**: ~$1.50 per million tokens (50% discount)

Monitor your usage at: https://console.anthropic.com/

## Troubleshooting

### API Key Not Found

**Error:** `ANTHROPIC_API_KEY environment variable not set!`

**Solution:**
1. Ensure `.env` file exists in project root
2. Verify the file contains: `ANTHROPIC_API_KEY=your_key_here`
3. Restart your Python interpreter to reload environment variables

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'langchain_anthropic'`

**Solution:**
```bash
pip install langchain-anthropic langchain-core python-dotenv
```

### JSON Parsing Errors

**Error:** `JSONDecodeError: Expecting value`

**Possible causes:**
- LLM returned text instead of JSON (rare with Claude)
- Network timeout interrupted response

**Solution:**
- Check your prompt engineering (ensure JSON-only instruction is clear)
- Increase timeout in config: `PipelineConfig.llm.TIMEOUT`
- Retry the extraction

### Rate Limiting

**Error:** `RateLimitError: Rate limit exceeded`

**Solution:**
- Wait a few seconds and retry
- The module automatically retries with exponential backoff
- Consider upgrading your Anthropic API tier

## Best Practices

1. **Use Temperature 0** for extraction tasks (consistency)
2. **Monitor token usage** to control costs
3. **Validate extracted JSON** using `validate_extraction_quality()`
4. **Review source_chunk IDs** for traceability
5. **Check confidence scores** - low confidence may indicate poor retrieval
6. **Test with various queries** before production use

## Next Steps

- Implement two-pass extraction for complex queries
- Add custom JSON schemas for specific document types
- Integrate with your application's workflow
- Set up monitoring and logging for production use

## Support

For issues or questions:
1. Check this README and code documentation
2. Review the comprehensive test output
3. Consult the strategic analysis document
4. Review LangChain and Anthropic documentation
