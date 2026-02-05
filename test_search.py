import os
import sys

# Add the parent directory to sys.path to import the app
sys.path.append(os.getcwd())

from app.tasks import transcribe_and_embed_task, search_audio_task, ModelManager

def test_semantic_logic():
    print("--- Starting Semantic Search Logic Test ---")
    
    # Mocking a transcription result for testing indexing
    # In a real scenario, this comes from Whisper
    mock_id = "test_task_123"
    mock_audio_path = "mock_audio.wav" # Doesn't need to exist for this test since we mock the result
    
    # We will manually inject data into ChromaDB for testing
    collection = ModelManager.get_chroma_collection()
    embedding_model = ModelManager.get_embedding_model()
    
    documents = [
        "We need to cut down expenses by 20% to save some money.",
        "The weather in London is quite rainy today.",
        "The meeting about the budget is scheduled for tomorrow at 10 AM.",
        "Apples and oranges are healthy fruits."
    ]
    
    ids = [f"{mock_id}_{i}" for i in range(len(documents))]
    metadatas = [
        {"start": 0.0, "end": 5.0, "label": "finance"},
        {"start": 5.0, "end": 10.0, "label": "weather"},
        {"start": 10.0, "end": 15.0, "label": "finance"},
        {"start": 15.0, "end": 20.0, "label": "food"}
    ]
    
    print("1. Indexing mock documents...")
    embeddings = embedding_model.encode(documents).tolist()
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {len(documents)} chunks.")
    
    print("\n2. Testing Semantic Queries:")
    
    queries = [
        "money issues",       # Should match expenses/budget
        "what's the weather",  # Should match London rain
        "healthy eating",     # Should match apples/oranges
        "financial planning"   # Should match budget/expenses
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = search_audio_task(query, n_results=2)
        for i, res in enumerate(results):
            print(f"  [{i+1}] Result: '{res['text']}' (Score: {res['distance']:.4f})")

if __name__ == "__main__":
    test_semantic_logic()
