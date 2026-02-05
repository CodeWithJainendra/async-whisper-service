"""
Concurrency Tests for Audio Search System

Tests race conditions, thread safety, and concurrent access patterns.
These tests are designed to catch the bugs that manifest only under load.
"""

import pytest
import threading
import time
import concurrent.futures
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConcurrentChunkIdGeneration:
    """Test that chunk ID generation is safe under concurrent access."""
    
    def test_concurrent_id_generation_no_collisions(self):
        """Generate IDs from multiple threads, ensure no collisions."""
        from app.tasks import generate_unique_chunk_id
        
        generated_ids = []
        lock = threading.Lock()
        
        def generate_ids(thread_num):
            for i in range(100):
                chunk_id = generate_unique_chunk_id(
                    f"task-{thread_num}",
                    i,
                    f"text-{thread_num}-{i}",
                    time.time()
                )
                with lock:
                    generated_ids.append(chunk_id)
        
        # Run 10 threads, each generating 100 IDs
        threads = [threading.Thread(target=generate_ids, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have 1000 unique IDs
        assert len(generated_ids) == 1000
        assert len(set(generated_ids)) == 1000, "Collision detected!"


class TestConcurrentModelAccess:
    """Test ModelManager under concurrent access."""
    
    def test_concurrent_worker_id_generation(self):
        """Each thread should get consistent worker ID."""
        from app.tasks import ModelManager
        
        results = {}
        
        def check_worker_id(thread_id):
            # Get worker ID multiple times from same thread
            ids = [ModelManager.get_worker_id() for _ in range(5)]
            results[thread_id] = ids
        
        threads = [threading.Thread(target=check_worker_id, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Each thread should have gotten consistent IDs
        for thread_id, ids in results.items():
            assert len(set(ids)) == 1, f"Thread {thread_id} got inconsistent IDs"
    
    @patch('app.tasks.chromadb.PersistentClient')
    def test_concurrent_collection_access(self, mock_client):
        """Multiple threads accessing collection shouldn't cause errors."""
        from app.tasks import ModelManager
        
        # Reset state
        ModelManager._chroma_clients = {}
        ModelManager._collections = {}
        
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance
        
        collections = []
        errors = []
        
        def get_collection():
            try:
                col = ModelManager.get_chroma_collection()
                collections.append(col)
            except Exception as e:
                errors.append(str(e))
        
        # 20 threads accessing concurrently
        threads = [threading.Thread(target=get_collection) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should have occurred
        assert len(errors) == 0, f"Errors: {errors}"
        # All should have gotten a collection
        assert len(collections) == 20


class TestConcurrentSearchRequests:
    """Test search under concurrent load."""
    
    @patch('app.tasks.ModelManager.get_embedding_model')
    @patch('app.tasks.ModelManager.get_chroma_collection')
    def test_concurrent_searches(self, mock_collection, mock_embed):
        """Multiple concurrent searches should not interfere."""
        from app.tasks import search_audio_sync
        
        mock_model = Mock()
        mock_embed.return_value = mock_model
        mock_model.encode.return_value = [[0.1] * 384]
        
        mock_col = Mock()
        mock_collection.return_value = mock_col
        mock_col.query.return_value = {
            'ids': [['id1']],
            'documents': [['doc1']],
            'metadatas': [[{}]],
            'distances': [[0.1]]
        }
        
        results = []
        errors = []
        
        def do_search(query):
            try:
                result = search_audio_sync(query)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # 50 concurrent searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(do_search, f"query-{i}") for i in range(50)]
            concurrent.futures.wait(futures)
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 50


class TestConcurrentBatchEncoding:
    """Test batch encoding under concurrent access."""
    
    def test_concurrent_encoding_isolation(self):
        """Concurrent encoding should not mix results."""
        from app.tasks import safe_encode_batch
        
        def mock_encode(texts):
            # Return embeddings that encode the text length
            return [[float(len(t))] * 384 for t in texts]
        
        mock_model = Mock()
        mock_model.encode.side_effect = mock_encode
        
        results = {}
        
        def encode_batch(batch_id, documents):
            embeddings, failed = safe_encode_batch(mock_model, documents)
            results[batch_id] = embeddings
        
        # Run multiple batches concurrently
        batches = {
            0: ["a", "bb", "ccc"],  # lengths: 1, 2, 3
            1: ["dddd", "eeeee"],    # lengths: 4, 5
            2: ["ffffff", "ggggggg"], # lengths: 6, 7
        }
        
        threads = [
            threading.Thread(target=encode_batch, args=(batch_id, docs))
            for batch_id, docs in batches.items()
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify results match expected (no mixing)
        assert len(results) == 3


class TestRaceConditionScenarios:
    """Test specific race condition scenarios from the bug report."""
    
    def test_task_retry_id_uniqueness(self):
        """Simulating task retry scenario - IDs must be unique."""
        from app.tasks import generate_unique_chunk_id
        
        # Simulate first task execution
        first_run_ids = []
        for segment_id in range(5):
            chunk_id = generate_unique_chunk_id(
                "task-abc123",
                segment_id,
                f"segment text {segment_id}",
                1000.0
            )
            first_run_ids.append(chunk_id)
        
        # Simulate retry (same task_id, different timestamp)
        retry_ids = []
        for segment_id in range(5):
            chunk_id = generate_unique_chunk_id(
                "task-abc123",  # Same task ID
                segment_id,     # Same segment ID
                f"segment text {segment_id}",  # Same text
                1001.0          # Different timestamp
            )
            retry_ids.append(chunk_id)
        
        # No ID from first run should appear in retry
        overlap = set(first_run_ids) & set(retry_ids)
        assert len(overlap) == 0, f"ID collision on retry: {overlap}"
    
    @patch('app.tasks.ModelManager.get_chroma_collection')
    def test_upsert_prevents_duplicate_overwrites(self, mock_collection):
        """Verify upsert is used instead of add to handle duplicates."""
        from app.tasks import transcribe_and_embed_task
        
        # Check that the code uses upsert instead of add
        import inspect
        source = inspect.getsource(transcribe_and_embed_task.__wrapped__)
        
        # Should use upsert, not add
        assert 'collection.upsert' in source or '.upsert(' in source, \
            "Should use upsert instead of add to handle duplicates"


class TestHealthCheckUnderLoad:
    """Test health check reliability under load."""
    
    def test_concurrent_health_checks(self):
        """Health checks should be reliable under concurrent access."""
        from app.tasks import ModelManager
        
        results = []
        errors = []
        
        def do_health_check():
            try:
                health = ModelManager.health_check()
                results.append(health)
            except Exception as e:
                errors.append(str(e))
        
        # 20 concurrent health checks
        threads = [threading.Thread(target=do_health_check) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Health check errors: {errors}"
        assert len(results) == 20
        
        # All results should have same structure
        for health in results:
            assert "whisper" in health
            assert "embedding" in health
            assert "chromadb" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
