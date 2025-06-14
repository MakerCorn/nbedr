# nBEDR Prompt Templates

This directory contains prompt templates used by nBEDR for various operations. These templates can be customized to improve the quality and relevance of embeddings for your specific use case.

## Available Templates

### Embedding Templates

- **`embedding_prompt_template.txt`**: Default template for embedding generation
  - Used to provide context and instructions for generating document embeddings
  - Supports variables: `{content}`, `{document_type}`, `{metadata}`
  - Customize this to improve embedding quality for your domain

### Legacy RAFT Templates (from raft-toolkit)

The following templates are inherited from the RAFT toolkit and can be used for reference or adapted for embedding use cases:

- **`gpt_template.txt`**: GPT-style question-answering template
  - Used for generating structured answers with reasoning and citations
  - Format includes step-by-step reasoning and follow-up questions

- **`gpt_qa_template.txt`**: GPT question generation template
  - Used for generating synthetic questions from context
  - Includes content filtering and complexity guidelines

- **`llama_template.txt`**: Llama-style question-answering template
  - Optimized for Llama and similar models
  - Includes detailed reasoning and answer extraction format

- **`llama_qa_template.txt`**: Llama question generation template
  - Generates questions with varying complexity levels
  - Includes specific formatting instructions

## Customizing Templates

### For Embedding Generation

1. **Copy the default template**:
   ```bash
   cp embedding_prompt_template.txt my_custom_template.txt
   ```

2. **Edit the template** to include domain-specific instructions:
   ```
   Generate embeddings optimized for [your domain] documents.
   Focus on [specific concepts/terminology] relevant to [your use case].
   ```

3. **Configure nBEDR** to use your custom template:
   ```bash
   export EMBEDDING_PROMPT_TEMPLATE="/path/to/templates/my_custom_template.txt"
   ```

### Template Variables

When creating custom templates, you can use these variables:

- `{content}`: The actual document content to be embedded
- `{document_type}`: File type (pdf, txt, json, pptx, etc.)
- `{metadata}`: Additional document metadata (file size, source, etc.)
- `{chunk_index}`: Index of the current chunk within the document
- `{chunking_strategy}`: The chunking method used (semantic, fixed, sentence)

### Best Practices

1. **Be Specific**: Include domain-specific terminology and concepts
2. **Provide Context**: Explain the intended use case for the embeddings
3. **Keep It Concise**: Avoid overly long prompts that might confuse the model
4. **Test and Iterate**: Experiment with different prompts and measure embedding quality
5. **Consider Your Model**: Different embedding models may respond better to different prompt styles

## Configuration

Set the prompt template path in your environment:

```bash
# Use a custom embedding prompt template
export EMBEDDING_PROMPT_TEMPLATE="/path/to/templates/my_template.txt"

# Use default template (no configuration needed)
# nBEDR will use embedding_prompt_template.txt by default
```

Or configure via the nBEDR configuration:

```python
config = EmbeddingConfig(
    embedding_prompt_template="templates/my_custom_template.txt"
)
```

## Examples

### Medical Documents
```
Generate embeddings for medical literature that capture:
- Clinical terminology and procedures
- Drug names and dosages
- Symptoms and diagnoses
- Treatment protocols and outcomes

Content: {content}
```

### Legal Documents
```
Generate embeddings for legal documents focusing on:
- Legal terminology and concepts
- Case citations and precedents
- Statutory references
- Contractual terms and conditions

Document Type: {document_type}
Content: {content}
```

### Technical Documentation
```
Generate embeddings for technical documentation emphasizing:
- API endpoints and parameters
- Code examples and syntax
- Configuration options
- Error messages and troubleshooting

Content: {content}
Metadata: {metadata}
```