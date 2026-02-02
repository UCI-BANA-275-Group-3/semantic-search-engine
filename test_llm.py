from src.llm_enhancement import summarize_top_k

# 1. Mock some search results (similar to what your engine produces)
mock_results = [
    {
        "docid": "Paper_A",
        "text": "Dynamic capabilities refer to a firm's ability to integrate, build, and reconfigure internal competences."
    },
    {
        "docid": "Paper_B",
        "text": "The literature on dynamic capabilities emphasizes sensing opportunities and seizing them through investment."
    }
]

# 2. Run the function
query = "What are dynamic capabilities?"
print("--- TESTING SUMMARIZER ---")
output = summarize_top_k(query, mock_results)

# 3. Print result
print(output)
