import time
from clinxplain.rag.pipeline import RAGPipeline

def test_caching():
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    query = "What is the capital of France?"
    
    # First run - Cache Miss
    print(f"\n--- Query 1: '{query}' (Expect Cache Miss) ---")
    start = time.time()
    response1 = pipeline.query(query)
    end = time.time()
    print(f"Response: {response1}")
    print(f"Time: {end - start:.4f} seconds")
    
    # Second run - Cache Hit (Exact Match)
    print(f"\n--- Query 2: '{query}' (Expect Cache Hit - Exact) ---")
    start = time.time()
    response2 = pipeline.query(query)
    end = time.time()
    print(f"Response: {response2}")
    print(f"Time: {end - start:.4f} seconds")


    # Third run - Cache Hit (Semantic Match)
    query_semantic = "Can you tell me the capital of France?"
    print(f"\n--- Query 3: '{query_semantic}' (Expect Cache Hit - Semantic) ---")
    start = time.time()
    response3 = pipeline.query(query_semantic)
    end = time.time()
    print(f"Response: {response3}")
    print(f"Time: {end - start:.4f} seconds")

if __name__ == "__main__":
    test_caching()
