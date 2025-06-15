# Migration Guide - Legacy API to New Provider System

This guide helps you migrate from the legacy embedding client API to the new multi-provider system introduced in the latest version.

## 🎯 **Overview of Changes**

The new version introduces a comprehensive embedding provider system that supports 7 different providers while maintaining backward compatibility with the legacy API.

### **Key Benefits of New API**:
- **Multi-Provider Support**: OpenAI, Azure OpenAI, AWS Bedrock, Google Vertex AI, LMStudio, Ollama, Llama.cpp
- **Unified Interface**: Consistent API across all providers
- **Enhanced Configuration**: Flexible, validated configuration system
- **Better Error Handling**: Robust error handling with fallback behavior
- **Type Safety**: Complete MyPy compliance with proper type annotations
- **Security Hardening**: Comprehensive security fixes and best practices

## 🔄 **API Migration Examples**

### **1. Basic Embedding Generation**

#### **Before (Legacy API)**:
```python
from core.clients.openai_client import EmbeddingClient

# Create client
client = EmbeddingClient()

# Generate embeddings
texts = ["Hello world", "How are you?"]
embeddings = await client.generate_embeddings(texts)

# Use embeddings
for i, embedding in enumerate(embeddings):
    print(f"Text: {texts[i]}")
    print(f"Embedding dimensions: {len(embedding)}")
```

#### **After (New API)**:
```python
from core.clients import create_provider_from_config
from core.config import EmbeddingConfig

# Create configuration
config = EmbeddingConfig(
    provider="openai",
    api_key="your-api-key",
    model="text-embedding-3-small",
    dimensions=1536
)

# Create provider
provider = create_provider_from_config(config)

# Generate embeddings
texts = ["Hello world", "How are you?"]
result = await provider.generate_embeddings(texts)

# Use embeddings
for i, embedding in enumerate(result.embeddings):
    print(f"Text: {texts[i]}")
    print(f"Embedding dimensions: {len(embedding)}")
    
# Access additional information
print(f"Model used: {result.model}")
print(f"Token usage: {result.usage}")
```

### **2. Configuration Management**

#### **Before (Legacy API)**:
```python
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["EMBEDDING_MODEL"] = "text-embedding-ada-002"

# Create client (reads from environment)
client = EmbeddingClient()
```

#### **After (New API)**:
```python
from core.config import EmbeddingConfig, get_config

# Option 1: Direct configuration
config = EmbeddingConfig(
    provider="openai",
    api_key="your-api-key",
    model="text-embedding-3-small",
    dimensions=1536,
    batch_size=100,
    rate_limit_enabled=True
)

# Option 2: From environment variables
config = EmbeddingConfig.from_env()

# Option 3: Using get_config with overrides
config = get_config(
    provider="openai",
    model="text-embedding-3-large"
)
```

### **3. Multiple Provider Support**

#### **Before (Legacy API)**:
```python
# Only OpenAI was supported
from core.clients.openai_client import EmbeddingClient

client = EmbeddingClient()
embeddings = await client.generate_embeddings(texts)
```

#### **After (New API)**:
```python
from core.clients import create_provider_from_config
from core.config import EmbeddingConfig

# OpenAI
openai_config = EmbeddingConfig(provider="openai", api_key="key")
openai_provider = create_provider_from_config(openai_config)

# Azure OpenAI
azure_config = EmbeddingConfig(
    provider="azure_openai",
    api_key="key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="your-deployment"
)
azure_provider = create_provider_from_config(azure_config)

# AWS Bedrock
bedrock_config = EmbeddingConfig(
    provider="aws_bedrock",
    aws_region="us-east-1",
    model="amazon.titan-embed-text-v1"
)
bedrock_provider = create_provider_from_config(bedrock_config)

# Local Ollama
ollama_config = EmbeddingConfig(
    provider="ollama",
    model="nomic-embed-text",
    ollama_base_url="http://localhost:11434"
)
ollama_provider = create_provider_from_config(ollama_config)
```

### **4. Error Handling and Fallbacks**

#### **Before (Legacy API)**:
```python
try:
    embeddings = await client.generate_embeddings(texts)
except Exception as e:
    print(f"Error: {e}")
    # Manual fallback handling
```

#### **After (New API)**:
```python
# Automatic fallback to mock embeddings on API failure
result = await provider.generate_embeddings(texts)

# Always returns valid EmbeddingResult, even on API failure
print(f"Generated {len(result.embeddings)} embeddings")
print(f"Model: {result.model}")

# Check if fallback was used
if result.usage.get("fallback_used"):
    print("Warning: Using mock embeddings due to API failure")
```

## 🔧 **Configuration Migration**

### **Environment Variables**

#### **Before**:
```bash
export OPENAI_API_KEY="your-key"
export EMBEDDING_MODEL="text-embedding-ada-002"
export EMBEDDING_DIMENSIONS="1536"
```

#### **After**:
```bash
export EMBEDDING_PROVIDER="openai"
export OPENAI_API_KEY="your-key"
export EMBEDDING_MODEL="text-embedding-3-small"
export EMBEDDING_DIMENSIONS="1536"
export BATCH_SIZE="100"
export RATE_LIMIT_ENABLED="true"
```

### **Configuration Files**

#### **Before (Limited)**:
```json
{
  "api_key": "your-key",
  "model": "text-embedding-ada-002"
}
```

#### **After (Comprehensive)**:
```json
{
  "provider": "openai",
  "api_key": "your-key",
  "model": "text-embedding-3-small",
  "dimensions": 1536,
  "batch_size": 100,
  "max_workers": 4,
  "rate_limit_enabled": true,
  "rate_limit_requests_per_minute": 500,
  "rate_limit_tokens_per_minute": 350000,
  "chunk_size": 512,
  "chunking_strategy": "semantic"
}
```

## 🚀 **Advanced Features Migration**

### **1. Rate Limiting**

#### **New Feature (Not Available Before)**:
```python
from core.config import EmbeddingConfig

config = EmbeddingConfig(
    provider="openai",
    api_key="your-key",
    rate_limit_enabled=True,
    rate_limit_requests_per_minute=500,
    rate_limit_tokens_per_minute=350000,
    rate_limit_strategy="sliding_window"
)

provider = create_provider_from_config(config)
```

### **2. Custom Embedding Prompts**

#### **New Feature (Not Available Before)**:
```python
# Set custom prompt template
config = EmbeddingConfig(
    provider="openai",
    api_key="your-key",
    embedding_prompt_template="templates/medical_template.txt",
    embedding_custom_variables={"domain": "healthcare"}
)
```

### **3. Multi-Instance Coordination**

#### **New Feature (Not Available Before)**:
```python
# Automatic coordination when running multiple instances
# No code changes needed - coordination is automatic
```

## 🔄 **Step-by-Step Migration Process**

### **Step 1: Update Dependencies**
```bash
# Update to latest version
pip install --upgrade nbedr

# Install with all providers
pip install nbedr[all]
```

### **Step 2: Update Imports**
```python
# Old imports (still work for backward compatibility)
from core.clients.openai_client import EmbeddingClient

# New imports (recommended)
from core.clients import create_provider_from_config
from core.config import EmbeddingConfig
```

### **Step 3: Update Configuration**
```python
# Replace direct client creation
# OLD:
client = EmbeddingClient()

# NEW:
config = EmbeddingConfig(provider="openai", api_key="your-key")
provider = create_provider_from_config(config)
```

### **Step 4: Update API Calls**
```python
# Replace embedding generation calls
# OLD:
embeddings = await client.generate_embeddings(texts)

# NEW:
result = await provider.generate_embeddings(texts)
embeddings = result.embeddings  # Extract embeddings if needed
```

### **Step 5: Update Error Handling**
```python
# Enhanced error handling with automatic fallbacks
try:
    result = await provider.generate_embeddings(texts)
    if result.usage.get("fallback_used"):
        logger.warning("Using mock embeddings due to API failure")
except Exception as e:
    logger.error(f"Embedding generation failed: {e}")
```

## 🧪 **Testing Migration**

### **Update Test Code**

#### **Before**:
```python
from unittest.mock import Mock
from core.clients.openai_client import EmbeddingClient

def test_embeddings():
    client = EmbeddingClient()
    # Test code...
```

#### **After**:
```python
from unittest.mock import Mock
from core.clients import create_provider_from_config
from core.config import EmbeddingConfig

def test_embeddings():
    config = EmbeddingConfig(provider="openai", api_key="test-key")
    provider = create_provider_from_config(config)
    # Test code...
```

## 🔍 **Backward Compatibility**

### **Legacy API Still Supported**
The legacy API continues to work without changes:

```python
# This still works exactly as before
from core.clients.openai_client import EmbeddingClient

client = EmbeddingClient()
embeddings = await client.generate_embeddings(texts)
```

### **Gradual Migration**
You can migrate gradually:
1. **Phase 1**: Update dependencies, keep existing code
2. **Phase 2**: Update configuration management
3. **Phase 3**: Update to new provider API
4. **Phase 4**: Add new features (rate limiting, multiple providers)

## 🎯 **Migration Checklist**

- [ ] **Update Dependencies**: Install latest nbedr version
- [ ] **Update Imports**: Add new provider system imports
- [ ] **Update Configuration**: Migrate to new EmbeddingConfig
- [ ] **Update API Calls**: Use new provider.generate_embeddings()
- [ ] **Update Error Handling**: Leverage automatic fallbacks
- [ ] **Update Tests**: Use new test fixtures and patterns
- [ ] **Update Environment Variables**: Add new provider-specific variables
- [ ] **Test Migration**: Verify functionality with new API
- [ ] **Add New Features**: Implement rate limiting, multiple providers as needed

## 🆘 **Troubleshooting**

### **Common Migration Issues**

#### **Import Errors**:
```python
# If you get import errors, ensure you have the latest version
pip install --upgrade nbedr

# Check version
python -c "import nbedr; print(nbedr.__version__)"
```

#### **Configuration Errors**:
```python
# Validate your configuration
config = EmbeddingConfig(provider="openai", api_key="your-key")
print(config.to_dict())  # Check configuration
```

#### **API Compatibility**:
```python
# Both APIs work side by side
from core.clients.openai_client import EmbeddingClient  # Legacy
from core.clients import create_provider_from_config    # New

# Use whichever you prefer
```

## 📚 **Additional Resources**

- **API Documentation**: See updated README.md for comprehensive examples
- **Security Guide**: See docs/SECURITY_FIXES.md for security improvements
- **Test Examples**: See tests/ directory for updated test patterns
- **Configuration Reference**: See core/config.py for all configuration options

---

**Need Help?** The migration is designed to be gradual and backward-compatible. You can migrate at your own pace while taking advantage of new features as needed.
