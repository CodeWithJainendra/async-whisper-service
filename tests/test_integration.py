"""
Integration Tests for Audio Search API

Tests end-to-end functionality of the API endpoints.
"""

import pytest
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def app():
    """Create test Flask application."""
    from app import create_app
    
    app = create_app()
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_status(self, client):
        """Health endpoint should return status dict."""
        response = client.get('/health')
        
        assert response.status_code in [200, 503]
        data = json.loads(response.data)
        
        assert 'status' in data
        assert 'whisper' in data
        assert 'embedding' in data
        assert 'chromadb' in data


class TestTranscribeEndpoint:
    """Tests for /asr endpoint."""
    
    def test_missing_audio_file(self, client):
        """Should return error when no audio file provided."""
        response = client.post('/asr')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error_code' in data
    
    def test_empty_filename(self, client):
        """Should reject empty filename."""
        from io import BytesIO
        
        response = client.post('/asr', data={
            'audio': (BytesIO(b'fake audio'), '')
        })
        
        assert response.status_code == 400


class TestSearchEndpoint:
    """Tests for /search endpoint."""
    
    def test_missing_query(self, client):
        """Should return error when query missing."""
        response = client.get('/search')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error_code' in data
        assert data['error_code'] == 'MISSING_QUERY'
    
    def test_query_too_short(self, client):
        """Should reject very short queries."""
        response = client.get('/search?q=a')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == 'QUERY_TOO_SHORT'
    
    def test_search_returns_results_structure(self, client):
        """Search should return proper response structure."""
        response = client.get('/search?q=test+query')
        
        # May fail if model not loaded, but should not error
        data = json.loads(response.data)
        assert 'success' in data
        if data['success']:
            assert 'results' in data
            assert 'query' in data


class TestAnalyzeEndpoint:
    """Tests for /analyze/<task_id> endpoint."""
    
    def test_missing_task_id(self, client):
        """Should handle invalid task ID."""
        response = client.post('/analyze/nonexistent-task-id')
        
        # Should return 404 for pending/nonexistent task
        assert response.status_code in [400, 404]


class TestTaskStatusEndpoint:
    """Tests for /tasks/<task_id> endpoint."""
    
    def test_pending_task_status(self, client):
        """Should return proper status for pending task."""
        response = client.get('/tasks/fake-task-id-123')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['task_id'] == 'fake-task-id-123'
        assert 'status' in data


class TestStatsEndpoint:
    """Tests for /stats endpoint."""
    
    def test_stats_returns_counts(self, client):
        """Stats endpoint should return model and index stats."""
        response = client.get('/stats')
        
        data = json.loads(response.data)
        assert 'success' in data
        if data['success']:
            assert 'indexed_chunks' in data
            assert 'models' in data


class TestErrorHandling:
    """Test error handling across endpoints."""
    
    def test_404_for_unknown_routes(self, client):
        """Unknown routes should return 404."""
        response = client.get('/unknown/route')
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Wrong HTTP method should return 405."""
        response = client.get('/asr')  # Should be POST
        assert response.status_code == 405


class TestResponseFormats:
    """Test response format consistency."""
    
    def test_all_errors_have_error_code(self, client):
        """All error responses should include error_code."""
        # Test various error scenarios
        responses = [
            client.post('/asr'),  # Missing file
            client.get('/search'),  # Missing query
        ]
        
        for response in responses:
            if response.status_code >= 400:
                data = json.loads(response.data)
                assert 'error_code' in data or 'error' in data, \
                    f"Error response missing error info: {data}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
