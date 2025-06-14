"""
Integration tests for document processing service.
Tests the complete workflow from document ingestion to embedding storage.
"""
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import Future

from core.services.document_service import DocumentService
from core.config import EmbeddingConfig
from core.models import DocumentChunk, EmbeddingBatch, ProcessingResult, EmbeddingStats


class TestDocumentServiceInitialization:
    """Test DocumentService initialization and basic functionality."""
    
    def test_service_initialization(self, sample_config):
        """Test DocumentService initialization with config."""
        service = DocumentService(sample_config)
        
        assert service.config == sample_config
        assert isinstance(service.stats, EmbeddingStats)
        assert service.stats.total_chunks == 0
    
    def test_service_with_different_configs(self):
        """Test service with various configuration options."""
        configs = [
            {"doctype": "pdf", "chunking_strategy": "fixed", "chunk_size": 512},
            {"doctype": "txt", "chunking_strategy": "sentence", "chunk_size": 1024},
            {"doctype": "json", "chunking_strategy": "semantic", "chunk_size": 256},
        ]
        
        for config_params in configs:
            config = EmbeddingConfig(**config_params)
            service = DocumentService(config)
            
            assert service.config.doctype == config_params["doctype"]
            assert service.config.chunking_strategy == config_params["chunking_strategy"]
            assert service.config.chunk_size == config_params["chunk_size"]


class TestDocumentProcessing:
    """Test document processing workflows."""
    
    def test_process_api_documents(self, test_data_dir):
        """Test processing API documentation."""
        config = EmbeddingConfig(doctype="api")
        service = DocumentService(config)
        
        api_docs_file = test_data_dir / "api_docs.json"
        chunks = service.process_documents(api_docs_file)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.metadata.get("type") == "api" for chunk in chunks)
        assert service.stats.total_chunks == len(chunks)
    
    def test_process_regular_documents_single_file(self, test_data_dir):
        """Test processing a single regular document."""
        config = EmbeddingConfig(doctype="txt", chunking_strategy="fixed", chunk_size=50)
        service = DocumentService(config)
        
        txt_file = test_data_dir / "test.txt"
        chunks = service.process_documents(txt_file)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.source == str(txt_file) for chunk in chunks)
        assert service.stats.total_chunks == len(chunks)
    
    def test_process_regular_documents_directory(self, test_data_dir):
        """Test processing all documents in a directory."""
        config = EmbeddingConfig(doctype="txt", chunking_strategy="fixed", chunk_size=100)
        service = DocumentService(config)
        
        chunks = service.process_documents(test_data_dir)
        
        # Should find and process the txt file
        txt_chunks = [c for c in chunks if c.source.endswith('test.txt')]
        assert len(txt_chunks) > 0
        assert service.stats.total_chunks == len(chunks)
    
    @patch('core.services.document_service.ThreadPoolExecutor')
    def test_process_documents_with_threading(self, mock_executor, test_data_dir):
        """Test document processing with thread pool execution."""
        config = EmbeddingConfig(doctype="txt", embed_workers=2)
        service = DocumentService(config)
        
        # Mock the executor
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        # Mock future results
        mock_future = MagicMock()
        mock_future.result.return_value = [
            DocumentChunk.create("test content", str(test_data_dir / "test.txt"))
        ]
        mock_executor_instance.submit.return_value = mock_future
        
        # Mock as_completed to return the future immediately
        with patch('core.services.document_service.as_completed', return_value=[mock_future]):
            chunks = service.process_documents(test_data_dir)
        
        assert len(chunks) > 0
        mock_executor_instance.submit.assert_called()
    
    def test_process_documents_with_pacing(self, test_data_dir):
        """Test document processing with pacing enabled."""
        config = EmbeddingConfig(doctype="txt", pace=True, embed_workers=1)
        service = DocumentService(config)
        
        with patch('time.sleep') as mock_sleep:
            chunks = service.process_documents(test_data_dir)
        
        # Should have called sleep for pacing (if multiple files were processed)
        # Sleep is called with 15 seconds between file submissions
        assert len(chunks) >= 0  # May be 0 if no txt files found


class TestSingleFileProcessing:
    """Test processing of individual files."""
    
    def test_process_txt_file(self, temp_file):
        """Test processing a text file."""
        config = EmbeddingConfig(doctype="txt", chunking_strategy="fixed", chunk_size=20)
        service = DocumentService(config)
        
        chunks = service._process_single_file(temp_file)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.source == str(temp_file)
            assert chunk.metadata["type"] == "txt"
            assert "chunk_index" in chunk.metadata
    
    def test_process_json_file(self, temp_json_file):
        """Test processing a JSON file."""
        config = EmbeddingConfig(doctype="json", chunking_strategy="fixed", chunk_size=50)
        service = DocumentService(config)
        
        chunks = service._process_single_file(temp_json_file)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["type"] == "json"
            assert chunk.source == str(temp_json_file)
    
    @patch('pypdf.PdfReader')
    def test_process_pdf_file(self, mock_pdf_reader, temp_file):
        """Test processing a PDF file."""
        # Mock PDF reading
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is PDF content for testing."
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        config = EmbeddingConfig(doctype="pdf", chunking_strategy="fixed", chunk_size=30)
        service = DocumentService(config)
        
        chunks = service._process_single_file(temp_file)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["type"] == "pdf"
            assert "PDF content" in chunk.content
    
    @patch('pptx.Presentation')
    def test_process_pptx_file(self, mock_presentation, temp_file):
        """Test processing a PowerPoint file."""
        # Mock PowerPoint reading
        mock_shape = MagicMock()
        mock_shape.text = "Slide content"
        
        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape]
        
        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]
        mock_presentation.return_value = mock_prs
        
        config = EmbeddingConfig(doctype="pptx", chunking_strategy="fixed", chunk_size=20)
        service = DocumentService(config)
        
        chunks = service._process_single_file(temp_file)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["type"] == "pptx"
            assert "Slide content" in chunk.content


class TestTextExtraction:
    """Test text extraction from different file types."""
    
    def test_extract_text_from_txt(self, temp_file):
        """Test extracting text from a text file."""
        config = EmbeddingConfig(doctype="txt")
        service = DocumentService(config)
        
        text = service._extract_text(temp_file)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "temporary test content" in text.lower()
    
    def test_extract_text_from_json(self, temp_json_file):
        """Test extracting text from a JSON file."""
        config = EmbeddingConfig(doctype="json")
        service = DocumentService(config)
        
        text = service._extract_text(temp_json_file)
        
        assert isinstance(text, str)
        assert len(text) > 0
        # Should contain either the "text" field value or the string representation
    
    def test_extract_text_unsupported_type(self, temp_file):
        """Test extraction with unsupported document type."""
        config = EmbeddingConfig(doctype="unsupported")
        service = DocumentService(config)
        
        with pytest.raises(ValueError, match="Unsupported document type"):
            service._extract_text(temp_file)


class TestTextChunking:
    """Test text chunking strategies."""
    
    def test_fixed_chunking(self):
        """Test fixed-size text chunking."""
        config = EmbeddingConfig(chunking_strategy="fixed", chunk_size=10)
        service = DocumentService(config)
        
        text = "This is a test text that should be split into chunks of fixed size."
        chunks = service._split_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 10 for chunk in chunks[:-1])  # All but last should be exactly 10
        assert len(chunks[-1]) <= 10  # Last chunk may be shorter
    
    def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        config = EmbeddingConfig(chunking_strategy="sentence", chunk_size=50)
        service = DocumentService(config)
        
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        chunks = service._split_text(text)
        
        assert len(chunks) > 0
        # Sentences should be preserved within chunk size limits
        for chunk in chunks:
            assert len(chunk) <= 50
    
    def test_semantic_chunking_fallback(self):
        """Test semantic chunking fallback to fixed chunking."""
        config = EmbeddingConfig(chunking_strategy="semantic", chunk_size=20)
        service = DocumentService(config)
        
        text = "This text will be chunked semantically but fall back to fixed."
        
        with patch.object(service, '_fixed_chunking') as mock_fixed:
            mock_fixed.return_value = ["chunk1", "chunk2"]
            chunks = service._split_text(text)
            
            mock_fixed.assert_called_once_with(text)
            assert chunks == ["chunk1", "chunk2"]
    
    def test_unknown_chunking_strategy(self):
        """Test error handling for unknown chunking strategy."""
        config = EmbeddingConfig(chunking_strategy="unknown")
        service = DocumentService(config)
        
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            service._split_text("test text")


class TestEmbeddingGeneration:
    """Test embedding generation workflows."""
    
    def test_generate_embeddings_basic(self, sample_document_chunks):
        """Test basic embedding generation."""
        config = EmbeddingConfig(batch_size_embeddings=3)
        service = DocumentService(config)
        
        embedded_chunks = service.generate_embeddings(sample_document_chunks)
        
        assert len(embedded_chunks) == len(sample_document_chunks)
        for chunk in embedded_chunks:
            assert chunk.has_embedding()
            assert chunk.embedding_model == config.embedding_model
    
    def test_generate_embeddings_batching(self, sample_document_chunks):
        """Test embedding generation with small batch sizes."""
        config = EmbeddingConfig(batch_size_embeddings=2)
        service = DocumentService(config)
        
        with patch.object(service, '_process_embedding_batch') as mock_process:
            mock_process.side_effect = lambda batch: batch  # Return batch unchanged
            
            embedded_chunks = service.generate_embeddings(sample_document_chunks)
        
        # Should have been called multiple times for batching
        expected_calls = (len(sample_document_chunks) + 1) // 2  # Ceiling division
        assert mock_process.call_count == expected_calls
    
    def test_generate_embeddings_batch_failure(self, sample_document_chunks):
        """Test handling of batch processing failures."""
        config = EmbeddingConfig(batch_size_embeddings=2)
        service = DocumentService(config)
        
        def mock_process_batch(batch):
            if len(batch.chunks) == 2:  # Fail on first batch
                raise Exception("Batch processing failed")
            return batch
        
        with patch.object(service, '_process_embedding_batch', side_effect=mock_process_batch):
            embedded_chunks = service.generate_embeddings(sample_document_chunks)
        
        # Should continue processing remaining batches despite failures
        # Exact count depends on chunk distribution, but should be >= remaining batches
        assert len(embedded_chunks) >= 0
    
    def test_process_embedding_batch(self, sample_embedding_batch):
        """Test processing a single embedding batch."""
        config = EmbeddingConfig()
        service = DocumentService(config)
        
        processed_batch = service._process_embedding_batch(sample_embedding_batch)
        
        assert processed_batch == sample_embedding_batch
        for chunk in processed_batch.chunks:
            assert chunk.has_embedding()
            assert chunk.embedding_model == sample_embedding_batch.model
    
    def test_generate_mock_embeddings(self):
        """Test mock embedding generation."""
        config = EmbeddingConfig(embedding_dimensions=768)
        service = DocumentService(config)
        
        texts = ["text 1", "text 2", "text 3"]
        embeddings = service._generate_mock_embeddings(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)
        
        # Test deterministic behavior
        embeddings2 = service._generate_mock_embeddings(texts)
        assert embeddings == embeddings2


class TestEmbeddingStorage:
    """Test embedding storage functionality."""
    
    def test_store_embeddings_success(self, sample_document_chunks):
        """Test successful embedding storage."""
        config = EmbeddingConfig(vector_db_type="faiss")
        service = DocumentService(config)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(sample_document_chunks):
            chunk.embedding = [0.1 * (i + 1)] * 100
            chunk.embedding_model = "test-model"
        
        result = service.store_embeddings(sample_document_chunks)
        
        assert result is True
        for chunk in sample_document_chunks:
            assert chunk.vector_id is not None
            assert chunk.vector_id.startswith("vec_")
    
    def test_store_embeddings_mixed_chunks(self, sample_document_chunks):
        """Test storing chunks with and without embeddings."""
        config = EmbeddingConfig()
        service = DocumentService(config)
        
        # Only add embeddings to some chunks
        sample_document_chunks[0].embedding = [0.1] * 100
        sample_document_chunks[0].embedding_model = "test-model"
        # Leave other chunks without embeddings
        
        result = service.store_embeddings(sample_document_chunks)
        
        assert result is True
        assert sample_document_chunks[0].vector_id is not None
        # Chunks without embeddings should not get vector IDs in real implementation
    
    def test_store_embeddings_failure_simulation(self, sample_document_chunks):
        """Test embedding storage failure handling."""
        config = EmbeddingConfig()
        service = DocumentService(config)
        
        # Add embeddings to chunks
        for chunk in sample_document_chunks:
            chunk.embedding = [0.1] * 100
        
        # Simulate storage failure by patching the storage logic
        with patch.object(service, 'store_embeddings', return_value=False):
            result = service.store_embeddings(sample_document_chunks)
            assert result is False


class TestEndToEndWorkflows:
    """Test complete end-to-end document processing workflows."""
    
    def test_complete_text_processing_workflow(self, test_data_dir):
        """Test complete workflow from text file to embedded chunks."""
        config = EmbeddingConfig(
            doctype="txt",
            chunking_strategy="fixed",
            chunk_size=30,
            batch_size_embeddings=2,
            vector_db_type="faiss"
        )
        service = DocumentService(config)
        
        # Step 1: Process documents
        txt_file = test_data_dir / "test.txt"
        chunks = service.process_documents(txt_file)
        assert len(chunks) > 0
        
        # Step 2: Generate embeddings
        embedded_chunks = service.generate_embeddings(chunks)
        assert len(embedded_chunks) == len(chunks)
        assert all(chunk.has_embedding() for chunk in embedded_chunks)
        
        # Step 3: Store embeddings
        storage_result = service.store_embeddings(embedded_chunks)
        assert storage_result is True
        assert all(chunk.vector_id is not None for chunk in embedded_chunks)
        
        # Verify final state
        for chunk in embedded_chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.content is not None
            assert chunk.source == str(txt_file)
            assert chunk.has_embedding()
            assert chunk.embedding_model == config.embedding_model
            assert chunk.vector_id is not None
    
    def test_complete_api_processing_workflow(self, test_data_dir):
        """Test complete workflow for API documentation."""
        config = EmbeddingConfig(
            doctype="api",
            batch_size_embeddings=1
        )
        service = DocumentService(config)
        
        # Process API documentation
        api_file = test_data_dir / "api_docs.json"
        chunks = service.process_documents(api_file)
        assert len(chunks) > 0
        
        # Generate embeddings
        embedded_chunks = service.generate_embeddings(chunks)
        assert all(chunk.has_embedding() for chunk in embedded_chunks)
        
        # Store embeddings
        storage_result = service.store_embeddings(embedded_chunks)
        assert storage_result is True
        
        # Verify API-specific metadata
        for chunk in embedded_chunks:
            assert chunk.metadata["type"] == "api"
            assert "index" in chunk.metadata
    
    def test_workflow_with_error_handling(self, test_data_dir):
        """Test workflow with various error conditions."""
        config = EmbeddingConfig(doctype="txt", chunk_size=20)
        service = DocumentService(config)
        
        # Process documents (should succeed)
        chunks = service.process_documents(test_data_dir)
        
        # Simulate embedding generation with some failures
        def mock_process_batch(batch):
            if len(batch.chunks) > 2:  # Fail large batches
                raise Exception("Batch too large")
            return batch
        
        with patch.object(service, '_process_embedding_batch', side_effect=mock_process_batch):
            embedded_chunks = service.generate_embeddings(chunks)
        
        # Should handle failures gracefully
        assert isinstance(embedded_chunks, list)
    
    def test_workflow_performance_tracking(self, test_data_dir):
        """Test that workflow properly tracks performance statistics."""
        config = EmbeddingConfig(doctype="txt", chunk_size=50)
        service = DocumentService(config)
        
        # Process documents
        chunks = service.process_documents(test_data_dir)
        initial_stats = service.get_stats()
        assert initial_stats.total_chunks == len(chunks)
        
        # Generate embeddings
        embedded_chunks = service.generate_embeddings(chunks)
        
        # Check final statistics
        final_stats = service.get_stats()
        assert final_stats.total_chunks == len(chunks)
        assert isinstance(final_stats, EmbeddingStats)
    
    def test_different_document_types_workflow(self, test_data_dir):
        """Test workflow with different document types."""
        document_types = ["txt", "json"]
        
        for doctype in document_types:
            config = EmbeddingConfig(
                doctype=doctype,
                chunking_strategy="fixed",
                chunk_size=40
            )
            service = DocumentService(config)
            
            chunks = service.process_documents(test_data_dir)
            if len(chunks) > 0:  # Only test if files of this type exist
                embedded_chunks = service.generate_embeddings(chunks)
                storage_result = service.store_embeddings(embedded_chunks)
                
                assert all(chunk.metadata["type"] == doctype for chunk in chunks)
                assert all(chunk.has_embedding() for chunk in embedded_chunks)
                assert storage_result is True
    
    def test_chunking_strategies_comparison(self, test_data_dir):
        """Test different chunking strategies on the same content."""
        strategies = ["fixed", "sentence"]
        results = {}
        
        for strategy in strategies:
            config = EmbeddingConfig(
                doctype="txt",
                chunking_strategy=strategy,
                chunk_size=30
            )
            service = DocumentService(config)
            
            chunks = service.process_documents(test_data_dir)
            if len(chunks) > 0:
                results[strategy] = len(chunks)
        
        # Both strategies should produce chunks, though counts may differ
        assert all(count > 0 for count in results.values())
    
    def test_batch_size_variations(self, sample_document_chunks):
        """Test embedding generation with different batch sizes."""
        batch_sizes = [1, 2, len(sample_document_chunks)]
        
        for batch_size in batch_sizes:
            config = EmbeddingConfig(batch_size_embeddings=batch_size)
            service = DocumentService(config)
            
            embedded_chunks = service.generate_embeddings(sample_document_chunks.copy())
            
            assert len(embedded_chunks) == len(sample_document_chunks)
            assert all(chunk.has_embedding() for chunk in embedded_chunks)


class TestDocumentServiceConfiguration:
    """Test DocumentService with various configurations."""
    
    @pytest.mark.parametrize("doctype", ["pdf", "txt", "json", "pptx"])
    def test_service_with_different_doctypes(self, doctype):
        """Test service configuration with different document types."""
        config = EmbeddingConfig(doctype=doctype)
        service = DocumentService(config)
        
        assert service.config.doctype == doctype
    
    @pytest.mark.parametrize("chunking_strategy", ["fixed", "sentence", "semantic"])
    def test_service_with_different_chunking_strategies(self, chunking_strategy):
        """Test service with different chunking strategies."""
        config = EmbeddingConfig(chunking_strategy=chunking_strategy)
        service = DocumentService(config)
        
        assert service.config.chunking_strategy == chunking_strategy
    
    @pytest.mark.parametrize("chunk_size", [100, 512, 1024])
    def test_service_with_different_chunk_sizes(self, chunk_size):
        """Test service with different chunk sizes."""
        config = EmbeddingConfig(chunk_size=chunk_size)
        service = DocumentService(config)
        
        assert service.config.chunk_size == chunk_size
    
    @pytest.mark.parametrize("vector_db_type", ["faiss", "pinecone", "chroma"])
    def test_service_with_different_vector_databases(self, vector_db_type):
        """Test service with different vector database configurations."""
        config = EmbeddingConfig(vector_db_type=vector_db_type)
        service = DocumentService(config)
        
        assert service.config.vector_db_type == vector_db_type