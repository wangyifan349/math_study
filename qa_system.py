from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a library of questions and answers using basic biological knowledge
qa_pairs = {
    "What is a paramecium?": "A paramecium is a single-celled organism found in freshwater.",
    "Describe the structure of DNA.": "DNA is a double-helix structure composed of nucleotides.",
    "What is photosynthesis?": "Photosynthesis is the process by which green plants make their own food using sunlight.",
    "What does RNA do?": "RNA is responsible for coding, decoding, regulation, and expression of genes.",
    "What is an ecosystem?": "An ecosystem is a community of interacting organisms and their environment."
}
# Extract questions as a list for vectorization
questions = list(qa_pairs.keys())
# Vectorize the questions for cosine similarity calculations
vectorizer = CountVectorizer()
tf_matrix = vectorizer.fit_transform(questions)
# Inform the user about the system
print("This is a simple biological Q&A system. Type 'exit' to quit.")
while True:
    # Get user query
    query = input("You: ")
    if query.lower() == "exit":
        # If user types 'exit', break the loop
        print("Exiting the Q&A system. Goodbye!")
        break
    # Transform the query into the same vector space
    query_vector = vectorizer.transform([query])
    # Compute cosine similarities between the query and all questions
    cosine_similarities = cosine_similarity(query_vector, tf_matrix)
    # Flatten the similarities array for ease of processing
    similarities = cosine_similarities.flatten()
    # Find index of the maximum similarity
    most_similar_question_index = similarities.argmax()
    # Retrieve the most similar question and associated response
    similar_question = questions[most_similar_question_index]
    response = qa_pairs[similar_question]
    # Output all cosine similarity scores for informational purposes
    print("\nSimilarity scores:")
    for i, question in enumerate(questions):
        print(f"  '{question}': {similarities[i]:.4f}")
    # Output the response based on the highest similarity
    print(f"\nBot (similar to '{similar_question}'):")
    print(response)
    print("\n" + "-"*50)
