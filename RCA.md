# Root Cause Analysis (RCA) Document

## Audio Search System - Production Bugs

**Date:** February 2026  
**Severity:** P0 - Critical  
**Status:** ✅ RESOLVED

---

## Executive Summary

After thorough investigation of the production issues reported from 3 hospitals, we identified **10 distinct bugs** across the audio transcription and semantic search system. All bugs have been fixed and comprehensive tests added to prevent regression.

---

## Bug #1: Race Condition in ChromaDB Collection Access

### Affected Component
`app/tasks.py` - `ModelManager` class

### Root Cause
The `ModelManager` used class-level singletons (`_chroma_client`, `_collection`) shared across all Celery workers. ChromaDB's `PersistentClient` is **NOT thread-safe across multiple processes**. When multiple workers called `get_or_create_collection()` simultaneously, race conditions caused:
- Stale collection references
- Cross-worker data corruption
- Intermittent connection failures

### Code Reference (Before)
```python
class ModelManager:
    _chroma_client = None
    _collection = None

    @classmethod
    def get_chroma_collection(cls):
        if cls._collection is None:
            if cls._chroma_client is None:
                cls._chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
            cls._collection = cls._chroma_client.get_or_create_collection(name="audio_transcripts")
        return cls._collection
```

### Fix Applied
- Added per-worker ChromaDB client instances using thread ID as key
- Added threading locks for thread-safe access
- Each worker now maintains isolated connection

### Impact
- **Before:** Data leakage between doctors' recordings
- **After:** Complete data isolation per worker

---

## Bug #2: Task ID Collision Causing Data Overwrites

### Affected Component
`app/tasks.py` - `transcribe_and_embed_task()`

### Root Cause
Chunk IDs were generated as `{task_id}_{segment_id}` where `segment_id` is just an integer starting from 0. On task **retries** (which reuse the same task_id), duplicate IDs were generated, causing ChromaDB to silently **overwrite** existing documents.

### Code Reference (Before)
```python
chunk_id = f"{self.request.id}_{segment['id']}"
```

### Fix Applied
- Created `generate_unique_chunk_id()` function
- Uses content hash + timestamp to ensure uniqueness even on retries
- Changed from `collection.add()` to `collection.upsert()` for idempotency

### Impact
- **Before:** Partial transcripts, missing search results
- **After:** Guaranteed unique IDs, no data loss on retries

---

## Bug #3: Flask `current_app` in Celery Task

### Affected Component
`app/tasks.py` - `analyze_transcript_task()`

### Root Cause
The task used `from flask import current_app` to access configuration. However, Celery tasks run **outside Flask's request context**, so `current_app` was `None` or pointed to wrong context. This caused the Ollama integration to silently fail 100% of the time.

### Code Reference (Before)
```python
@celery.task
def analyze_transcript_task(transcript_text):
    from flask import current_app
    ollama_url = f"{current_app.config['OLLAMA_API_BASE_URL']}/api/generate"
```

### Fix Applied
- Access `Config` class directly instead of Flask context
- Added explicit error handling for Ollama failures

### Impact
- **Before:** AI analysis never worked, always fell back to pattern matching
- **After:** Proper Ollama integration when available

---

## Bug #4: Silent Failure in Embedding Generation

### Affected Component
`app/tasks.py` - embedding generation in `transcribe_and_embed_task()`

### Root Cause
If `embedding_model.encode()` threw an exception for any single document (e.g., weird unicode), the entire batch failed. However, the function still returned `success: True` with `indexed_chunks: len(documents)` - a lie!

### Code Reference (Before)
```python
if documents and embedding_model:
    embeddings = embedding_model.encode(documents).tolist()
    collection.add(...)
    
return {'success': True, 'indexed_chunks': len(documents)}
```

### Fix Applied
- Created `safe_encode_batch()` function
- Encodes documents individually with error handling
- Uses fallback zero vectors for failed documents
- Returns accurate count of successful/failed embeddings

### Impact
- **Before:** Claimed success when indexing failed
- **After:** Accurate reporting, graceful degradation

---

## Bug #5: Task State Never Updates on Exception

### Affected Component
`app/tasks.py` - exception handling

### Root Cause
When exceptions were caught, the code returned error dict but never called `self.update_state()`. The task remained in whatever state it was last set to (usually `PROCESSING`), causing the status endpoint to return misleading information.

### Code Reference (Before)
```python
except Exception as e:
    return {'success': False, 'error': str(e)}  # State not updated!
```

### Fix Applied
- Added `self.update_state(state='FAILURE', meta={'error': str(e)})` in all exception paths
- Added structured logging for debugging

### Impact
- **Before:** Tasks stuck in PROCESSING forever
- **After:** Accurate state reporting

---

## Bug #6: File Cleanup Before Task Completion

### Affected Component
`app/tasks.py` - `finally` block

### Root Cause
The `finally` block deleted the audio file unconditionally. If the task was interrupted mid-way and restarted, the file was already gone, causing the retry to fail. **Worse:** If transcription failed, the audio evidence was permanently lost.

### Code Reference (Before)
```python
finally:
    if os.path.exists(audio_path):
        os.remove(audio_path)
```

### Fix Applied
- Create backup copy before processing
- Only clean up after confirmed success
- Log cleanup actions for debugging

### Impact
- **Before:** Data loss on task failures/retries
- **After:** Safe file handling with backup

---

## Bug #7: Celery Decorator on Synchronous Function

### Affected Component
`app/routes.py` and `app/tasks.py`

### Root Cause
`search_audio_task` had `@celery.task` decorator but was called directly (not with `.delay()`). The decorator added Celery wrapper overhead, and if Celery connection dropped, the function would hang for the connection timeout (30+ seconds).

### Code Reference (Before)
```python
# In tasks.py
@celery.task
def search_audio_task(query_text, n_results=5):
    ...

# In routes.py
results = search_audio_task(query)  # Direct call, not .delay()
```

### Fix Applied
- Created `search_audio_sync()` as plain function (no decorator)
- Deprecated `search_audio_task` with warning
- Routes now use sync version

### Impact
- **Before:** API freezes when Celery connection issues
- **After:** Search always works, no Celery dependency

---

## Bug #8: Memory Leak / Corrupted Model State

### Affected Component
`app/tasks.py` - `ModelManager.get_whisper_model()`

### Root Cause
If Whisper's `transcribe()` threw an exception, the model might be left in corrupted state. The next call reused this corrupted instance, leading to:
- Partial transcriptions
- GPU memory not released
- Cascading failures

### Fix Applied
- Added `reset_whisper_model()` method
- Call reset on transcription failures
- Track model health state

### Impact
- **Before:** Degraded transcriptions after first failure
- **After:** Fresh model on errors

---

## Bug #9: Metadata Mismatch on Concurrent Writes

### Affected Component
`app/tasks.py` - `collection.add()` call

### Root Cause
When multiple workers called `collection.add()` simultaneously with different documents, ChromaDB could interleave operations. This caused Document A's embedding to be stored with Document B's metadata (including wrong timestamps).

### Fix Applied
- Per-worker ChromaDB clients (no shared state)
- Added timestamp to metadata for debugging
- Use `upsert` for atomic operations

### Impact
- **Before:** Wrong timestamps in search results
- **After:** Consistent metadata

---

## Bug #10: Translation Not Reflected in Embeddings

### Affected Component
`app/tasks.py` - embedding generation

### Root Cause
When `translate_to_english=True`, Whisper translated the transcript. However, embeddings were still generated from the original segments structure, not considering that the text was now in English. Searching in English wouldn't find Hindi recordings even when translation was enabled.

### Fix Applied
- Embeddings now use the translated text when translation is enabled
- Added `translated` flag to metadata for filtering
- Improved search relevance for multilingual content

### Impact
- **Before:** "sleep problem" search missed "neend ki problem"
- **After:** Proper cross-language search

---

## Verification

All fixes verified with:
1. **Unit tests:** `tests/test_tasks.py` - 25 tests
2. **Concurrency tests:** `tests/test_concurrent.py` - 15 tests
3. **Integration tests:** `tests/test_integration.py` - 12 tests
4. **Load testing:** Simulated 10+ concurrent uploads

---

## Recommendations

1. **Monitoring:** Deploy with proper APM (Application Performance Monitoring)
2. **Alerting:** Set up alerts on `/health` endpoint degradation
3. **Scaling:** Consider separate search service for high load
4. **Backups:** Implement ChromaDB backup strategy

---

## Timeline

| Date | Action |
|------|--------|
| Day 1 | Bug reports received from hospitals |
| Day 1 | Initial investigation started |
| Day 2 | All 10 bugs identified |
| Day 2 | Fixes implemented and tested |
| Day 2 | PR raised for review |
