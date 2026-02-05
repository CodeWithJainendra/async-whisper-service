# 🔥 Critical Production Bug Report: Audio Search System Corruption

## Priority: P0 - BLOCKING PRODUCTION DEPLOYMENT

---

## 📋 Executive Summary

Our Doctors Audio Search API was deployed to 3 hospitals as a pilot. After 2 weeks of usage, we're receiving **critical bug reports** from doctors. The system appears to have multiple interrelated issues causing:

- **Data leakage** between different doctors' recordings
- **Silent data loss** during indexing
- **Phantom task states** that never complete
- **Metadata corruption** in search results
- **Intermittent system freezes** under load

The CTO has mandated a **complete audit and fix** before we can proceed with the national rollout.

---

## 🚨 User Reports (Verbatim from Support Tickets)

### Ticket #1247 - Dr. Sharma (Cardiology, AIIMS Delhi)
> "I recorded a detailed patient consultation about mitral valve replacement. When I search for 'mitral valve', I get results from some other doctor's recording about diabetes! This is a HIPAA-level privacy breach!"

### Ticket #1251 - Dr. Patel (Orthopedics, Apollo Mumbai)
> "I uploaded 5 audio recordings yesterday. Today when I search, only 2 show up. The other 3 seem to have vanished. But when I check task status, it says SUCCESS for all of them!"

### Ticket #1256 - Dr. Rao (General Medicine, CMC Vellore)
> "Sometimes when I upload a file, the status shows 'PROCESSING' forever. I've waited 30 minutes. When I restart the app, it's still PROCESSING. Is my recording lost?"

### Ticket #1263 - IT Admin, Fortis Bangalore
> "Under heavy load (morning OPD hours when 15+ doctors upload simultaneously), the entire search API freezes for 2-3 minutes. We have to restart the Flask server."

### Ticket #1270 - Dr. Khanna (Psychiatry, NIMHANS)
> "I recorded in Hindi и asked my patient 'aapko neend mein problem hai?' When I search 'sleep problem' in English, nothing comes up. The translation feature doesn't seem to work properly with search."

### Ticket #1275 - Dr. Singh (Surgery, PGI Chandigarh)  
> "I searched for a specific phrase and got correct results, but the timestamp shows 45:30 when my recording was only 10 minutes long! The audio snippet it returns is completely wrong."

---

## 🔍 Technical Investigation Required

You are tasked with:

1. **Reproducing** each reported issue
2. **Root cause analysis** for each bug
3. **Fixing** all identified issues
4. **Adding tests** to prevent regression
5. **Documenting** architectural improvements

---

## 🧪 Reproduction Environment

```bash
# Start Redis
redis-server

# Start Flask (Terminal 1)
python run.py

# Start Celery with multiple workers to expose concurrency bugs (Terminal 2)
celery -A app.celery worker --loglevel=info --concurrency=4

# Simulate load (Terminal 3) - Upload 10 files simultaneously
for i in {1..10}; do
  curl -X POST -F "audio=@test_audio.webm" http://localhost:5000/asr &
done
wait
```

---

## 📁 Files to Investigate

| File | Suspected Issues |
|------|------------------|
| `app/__init__.py` | Celery initialization, task discovery |
| `app/tasks.py` | Core async logic, embedding, ChromaDB |
| `app/routes.py` | API endpoints, task invocation |
| `app/config.py` | Configuration management |

---

## 🎯 Expected Deliverables

### 1. Root Cause Analysis Document (`RCA.md`)
For each bug found, document:
- **Bug ID & Title**
- **Affected Component**
- **Root Cause** (with code references)
- **Impact Assessment**
- **Fix Strategy**

### 2. Code Fixes
- All bugs must be fixed
- Fixes should be minimal and targeted
- No unnecessary refactoring
- Each fix should be a separate commit with clear message

### 3. Test Suite (`tests/`)
Create comprehensive tests:
- `tests/test_tasks.py` - Unit tests for task functions
- `tests/test_concurrent.py` - Concurrency/race condition tests
- `tests/test_integration.py` - End-to-end tests

### 4. Monitoring & Health Check
- Add `/health` endpoint that checks:
  - Redis connectivity
  - ChromaDB status
  - Whisper model loaded
  - Embedding model loaded
- Add structured logging for debugging

### 5. Architecture Documentation (`ARCHITECTURE.md`)
- Document the async flow
- Explain concurrency model
- List known limitations

---

## ⏰ Time Expectation

| Seniority | Expected Time |
|-----------|---------------|
| Junior Engineer | 8-12 hours |
| Mid-level Engineer | 4-6 hours |
| Senior Engineer | 2-3 hours |

---

## 🏆 Evaluation Criteria

| Criteria | Weight |
|----------|--------|
| All bugs identified | 25% |
| All bugs correctly fixed | 30% |
| Test coverage | 20% |
| Code quality & documentation | 15% |
| Performance improvements | 10% |

---

## 💡 Hints (For Interviewers Only - DO NOT SHARE)

<details>
<summary>Click to reveal hints - INTERVIEWERS ONLY</summary>

### Bug Categories to Look For:

1. **Concurrency Issues**
   - Race conditions in singleton pattern
   - ChromaDB not thread-safe across workers
   - Task ID collisions on retries

2. **Context Issues**
   - Flask `current_app` used in Celery task (wrong context)
   - Celery task decorator on function called synchronously

3. **Silent Failures**
   - Exception handling that returns success
   - Embedding generation failures not surfaced
   - Task state not updated on exception path

4. **Data Integrity**
   - Metadata mismatch during concurrent writes
   - Duplicate IDs overwriting documents
   - File cleanup before task completion

5. **Design Flaws**
   - Translation not considered in embedding
   - Model corruption on exceptions
   - Memory leaks in model loading

### Key Files & Lines:
- `tasks.py:11-44` - ModelManager singleton issues
- `tasks.py:46-114` - transcribe_and_embed_task bugs
- `tasks.py:147-199` - current_app context issue
- `routes.py:49` - search_audio_task sync call with decorator

</details>

---

## 📝 Submission Guidelines

1. Fork the repository
2. Create a branch named `fix/production-bugs-<your-name>`
3. Make atomic commits for each fix
4. Create a Pull Request with:
   - Summary of bugs found
   - Description of each fix
   - Test results
5. Ensure all tests pass before submitting

---

## ⚠️ Important Notes

- Do NOT contact the original developers
- Do NOT look at git history for clues
- Assume this is real production code
- Time yourself - we track solving time

---

**Good Luck! The hospitals are waiting for your fix.** 🏥
