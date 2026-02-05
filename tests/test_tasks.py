"""
Unit Tests for Tasks Module

Tests the core functionality of the audio transcription and embedding tasks.
Covers:
- ModelManager thread safety
- Unique ID generation
- Safe batch encoding
- Error handling paths
"""

import pytest
import threading
import time
import hashlib
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGenerateUniqueChunkId:
    """Tests for unique chunk ID generation."""
    
    def test_unique_id_format(self):
        """Chunk IDs should follow expected format."""
        from app.tasks import generate_unique_chunk_id
        
        chunk_id = generate_unique_chunk_id(
            task_id="task-123",
            segment_id=0,
            text="Hello world",
            timestamp=1234567890.0
        )
        
        # Should start with task_id and segment_id
        assert chunk_id.startswith("task-123_0_")
        # Should have hash suffix
        assert len(chunk_id.split("_")) == 3
    
    def test_same_inputs_same_output(self):
        """Same inputs should produce same ID (deterministic)."""
        from app.tasks import generate_unique_chunk_id
        
        id1 = generate_unique_chunk_id("task", 0, "text", 123.0)
        id2 = generate_unique_chunk_id("task", 0, "text", 123.0)
        
        assert id1 == id2
    
    def test_different_timestamp_different_id(self):
        """Different timestamps should produce different IDs."""
        from app.tasks import generate_unique_chunk_id
        
        id1 = generate_unique_chunk_id("task", 0, "text", 123.0)
        id2 = generate_unique_chunk_id("task", 0, "text", 124.0)
        
        assert id1 != id2
    
    def test_different_text_different_id(self):
        """Different text should produce different IDs."""
        from app.tasks import generate_unique_chunk_id
        
        id1 = generate_unique_chunk_id("task", 0, "text1", 123.0)
        id2 = generate_unique_chunk_id("task", 0, "text2", 123.0)
        
        assert id1 != id2
    
    def test_retry_scenario_different_id(self):
        """Simulating task retry with same task_id but different timestamp."""
        from app.tasks import generate_unique_chunk_id
        
        # First attempt
        id1 = generate_unique_chunk_id("task-retry-test", 0, "same text", 1000.0)
        
        # Retry attempt (same task_id, segment_id, text but different timestamp)
        id2 = generate_unique_chunk_id("task-retry-test", 0, "same text", 1001.0)
        
        # Should NOT collide
        assert id1 != id2


class TestSafeEncodeBatch:
    """Tests for safe batch encoding with error handling."""
    
    def test_successful_encoding(self):
        """Normal documents should encode successfully."""
        from app.tasks import safe_encode_batch
        
        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1] * 384]
        
        documents = ["Hello world", "Test document"]
        embeddings, failed = safe_encode_batch(mock_model, documents)
        
        assert len(embeddings) == 2
        assert len(failed) == 0
    
    def test_handles_encoding_failure(self):
        """Should handle individual document encoding failures."""
        from app.tasks import safe_encode_batch
        
        # Mock model that fails on second call
        mock_model = Mock()
        call_count = [0]
        
        def encode_side_effect(docs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("Encoding failed")
            return [[0.1] * 384]
        
        mock_model.encode.side_effect = encode_side_effect
        
        documents = ["doc1", "doc2", "doc3"]
        embeddings, failed = safe_encode_batch(mock_model, documents)
        
        # Should have 3 embeddings (with fallback for failed one)
        assert len(embeddings) == 3
        # Second document should be in failed list
        assert 1 in failed
    
    def test_handles_weird_unicode(self):
        """Should handle documents with problematic unicode."""
        from app.tasks import safe_encode_batch
        
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1] * 384]
        
        documents = ["Normal text", "Weird \x00 null \x1f chars", "emoji 🎉"]
        embeddings, failed = safe_encode_batch(mock_model, documents)
        
        assert len(embeddings) == 3
    
    def test_handles_very_short_documents(self):
        """Should handle very short documents with fallback."""
        from app.tasks import safe_encode_batch
        
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1] * 384]
        
        documents = ["a", "", "  "]
        embeddings, failed = safe_encode_batch(mock_model, documents)
        
        # Should not fail, uses "empty segment" fallback
        assert len(embeddings) == 3


class TestModelManagerThreadSafety:
    """Tests for ModelManager thread safety."""
    
    def test_get_worker_id_unique_per_thread(self):
        """Each thread should get a unique worker ID."""
        from app.tasks import ModelManager
        
        worker_ids = []
        
        def collect_worker_id():
            worker_ids.append(ModelManager.get_worker_id())
        
        threads = [threading.Thread(target=collect_worker_id) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All worker IDs should be unique
        assert len(set(worker_ids)) == 5
    
    def test_health_check_returns_dict(self):
        """Health check should return proper status dict."""
        from app.tasks import ModelManager
        
        health = ModelManager.health_check()
        
        assert isinstance(health, dict)
        assert "whisper" in health
        assert "embedding" in health
        assert "chromadb" in health


class TestModelManagerReset:
    """Tests for model reset functionality."""
    
    def test_reset_whisper_model(self):
        """Should be able to reset Whisper model."""
        from app.tasks import ModelManager
        
        # Reset and check state
        ModelManager.reset_whisper_model()
        
        assert ModelManager._whisper_model is None
        assert ModelManager._whisper_healthy is False


class TestTranscribeAndEmbedTask:
    """Tests for the main transcription task."""
    
    @patch('app.tasks.ModelManager.get_whisper_model')
    @patch('app.tasks.ModelManager.get_embedding_model')
    @patch('app.tasks.ModelManager.get_chroma_collection')
    def test_handles_missing_file(self, mock_collection, mock_embed, mock_whisper):
        """Should handle missing audio file gracefully."""
        from app.tasks import transcribe_and_embed_task
        
        # Create a mock task instance
        mock_self = Mock()
        mock_self.request.id = "test-task-id"
        mock_self.update_state = Mock()
        
        # Call with non-existent file
        result = transcribe_and_embed_task.__wrapped__(
            mock_self, 
            "/nonexistent/file.wav", 
            False
        )
        
        assert result['success'] is False
        assert 'not found' in result['error'].lower()
        # Should have updated state to FAILURE
        mock_self.update_state.assert_called()
    
    @patch('app.tasks.ModelManager.get_whisper_model')
    def test_handles_model_load_failure(self, mock_whisper):
        """Should handle Whisper model load failure."""
        from app.tasks import transcribe_and_embed_task
        import tempfile
        
        mock_whisper.side_effect = RuntimeError("GPU out of memory")
        
        mock_self = Mock()
        mock_self.request.id = "test-task-id"
        mock_self.update_state = Mock()
        
        # Create a temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            result = transcribe_and_embed_task.__wrapped__(
                mock_self, 
                temp_path, 
                False
            )
            
            assert result['success'] is False
            # Should update state to FAILURE
            mock_self.update_state.assert_called()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestSearchAudioSync:
    """Tests for synchronous search function."""
    
    @patch('app.tasks.ModelManager.get_embedding_model')
    @patch('app.tasks.ModelManager.get_chroma_collection')
    def test_returns_empty_when_no_model(self, mock_collection, mock_embed):
        """Should return empty list when embedding model unavailable."""
        from app.tasks import search_audio_sync
        
        mock_embed.return_value = None
        
        results = search_audio_sync("test query")
        
        assert results == []
    
    @patch('app.tasks.ModelManager.get_embedding_model')
    @patch('app.tasks.ModelManager.get_chroma_collection')
    def test_handles_search_exception(self, mock_collection, mock_embed):
        """Should handle search exceptions gracefully."""
        from app.tasks import search_audio_sync
        
        mock_model = Mock()
        mock_embed.return_value = mock_model
        mock_model.encode.side_effect = RuntimeError("Encode failed")
        
        results = search_audio_sync("test query")
        
        assert results == []
    
    @patch('app.tasks.ModelManager.get_embedding_model')
    @patch('app.tasks.ModelManager.get_chroma_collection')
    def test_formats_results_correctly(self, mock_collection, mock_embed):
        """Should format ChromaDB results properly."""
        from app.tasks import search_audio_sync
        
        mock_model = Mock()
        mock_embed.return_value = mock_model
        mock_model.encode.return_value = [[0.1] * 384]
        
        mock_col = Mock()
        mock_collection.return_value = mock_col
        mock_col.query.return_value = {
            'ids': [['id1', 'id2']],
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'val1'}, {'key': 'val2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = search_audio_sync("test query", n_results=2)
        
        assert len(results) == 2
        assert results[0]['chunk_id'] == 'id1'
        assert results[0]['text'] == 'doc1'
        assert results[0]['distance'] == 0.1


class TestAnalyzeTranscriptTask:
    """Tests for transcript analysis task."""
    
    def test_rejects_short_transcript(self):
        """Should reject transcripts that are too short."""
        from app.tasks import analyze_transcript_task
        
        mock_self = Mock()
        mock_self.update_state = Mock()
        
        result = analyze_transcript_task.__wrapped__(mock_self, "short")
        
        assert "error" in result
        assert "too short" in result["error"].lower()
    
    @patch('requests.post')
    def test_falls_back_to_pattern_matching(self, mock_post):
        """Should fall back to pattern matching when Ollama fails."""
        from app.tasks import analyze_transcript_task
        
        mock_self = Mock()
        mock_self.update_state = Mock()
        
        # Simulate Ollama connection failure
        mock_post.side_effect = Exception("Connection refused")
        
        transcript = "The patient needs to take medicine. Schedule follow up on Jan 15."
        result = analyze_transcript_task.__wrapped__(mock_self, transcript)
        
        assert "summary" in result
        assert "action_items" in result
        assert "entities" in result
    
    @patch('requests.post')
    def test_extracts_action_items(self, mock_post):
        """Should extract action items from transcript."""
        from app.tasks import analyze_transcript_task
        
        mock_self = Mock()
        mock_self.update_state = Mock()
        mock_post.side_effect = Exception("No Ollama")
        
        transcript = "I need to schedule a follow-up. The patient should take aspirin daily. We must check blood pressure next week."
        result = analyze_transcript_task.__wrapped__(mock_self, transcript)
        
        # Should find action items with keywords
        assert len(result["action_items"]) > 0
    
    @patch('requests.post')
    def test_extracts_date_entities(self, mock_post):
        """Should extract date entities from transcript."""
        from app.tasks import analyze_transcript_task
        
        mock_self = Mock()
        mock_self.update_state = Mock()
        mock_post.side_effect = Exception("No Ollama")
        
        transcript = "Appointment scheduled for 15/01/2024. Follow up on Feb 20. Patient came on 01-12-2023."
        result = analyze_transcript_task.__wrapped__(mock_self, transcript)
        
        # Should find date entities
        date_entities = [e for e in result["entities"] if e.get("type") == "DATE"]
        assert len(date_entities) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
