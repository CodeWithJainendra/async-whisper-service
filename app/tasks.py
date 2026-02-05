import whisper
import os
from . import celery
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

print("Loading Whisper model in worker...")
try:
    model = whisper.load_model("turbo")
    print("Turbo model loaded.")
except Exception as e:
    print(f"Error loading 'turbo' model: {e}")
    print("Falling back to 'base' model.")
    model = whisper.load_model("base")

print("Loading Embedding model...")
try:
    # Use a small, fast model for embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None

# Initialize ChromaDB (Persistent)
chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
collection = chroma_client.get_or_create_collection(name="audio_transcripts")

@celery.task(bind=True)
def transcribe_and_embed_task(self, audio_path, translate_to_english=False):
    """
    Background task to transcribe audio AND generate semantic embeddings.
    """
    try:
        args = {}
        if translate_to_english:
             args['task'] = 'translate'

        self.update_state(state='PROCESSING', meta={'status': 'Transcribing...'})

        result = model.transcribe(audio_path, **args)
        detected_language = result.get('language', 'unknown')
        full_text = result['text']
        
        # --- NEW: Semantic Processing ---
        if embedding_model:
            self.update_state(state='PROCESSING', meta={'status': 'Indexing for Search...'})
            
            # 1. Chunking: Split into meaningful segments (using whisper segments)
            segments = result.get('segments', [])
            
            ids = []
            documents = []
            metadatas = []
            
            for segment in segments:
                # Create a unique ID for each chunk
                chunk_id = f"{self.request.id}_{segment['id']}"
                text_chunk = segment['text'].strip()
                
                if not text_chunk: 
                    continue
                    
                ids.append(chunk_id)
                documents.append(text_chunk)
                metadatas.append({
                    "task_id": self.request.id,
                    "start": segment['start'],
                    "end": segment['end'],
                    "language": detected_language
                })
            
            # 2. Embedding & Indexing
            # We explicitly compute embeddings to be safe, or let chroma do it with default ef?
            # We already loaded SentenceTransformer, let's use it explicitly for control
            embeddings = embedding_model.encode(documents).tolist()
            
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
        return {
            'success': True,
            'transcript': full_text,
            'language': detected_language,
            'indexed_chunks': len(documents) if embedding_model else 0
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

@celery.task
def search_audio_task(query_text, n_results=5):
    """
    Task to search the vector database.
    Can be run synchronously or async.
    """
    if not embedding_model:
        return {'error': 'Embedding model not initialized'}
        
    query_embedding = embedding_model.encode([query_text]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    # Format results for easier consumption
    formatted_results = []
    
    # Chroma returns lists of lists (because you can query multiple at once)
    if results['ids']:
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
            
    return formatted_results
