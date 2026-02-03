from src.similarity_search import search_top_k

query = input("Enter your search query: ")
results = search_top_k(query, 5)

print("\nTop-5 retrieved chunks:")
for i, r in enumerate(results, 1):
    print(f"\n--- Chunk {i} ---")
    print(f"Score: {r.get('score'):.4f}")
    print(f"Doc ID: {r.get('doc_id')}")
    print(f"Chunk ID: {r.get('chunk_id')}")
    print(f"Text:\n{r.get('chunk_text') or r.get('text') or r.get('preview') or r.get('snippet')}")
