import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example user-item rating matrix
# Rows are users: U1, U2, U3, ...
# Columns are items: I1, I2, I3, ...
# Values are ratings: 1-5, 0 means not rated
ratings = np.array([
    [4, 0, 0, 5, 1, 0, 0],
    [5, 5, 4, 0, 0, 0, 0],
    [0, 0, 0, 2, 4, 5, 0],
    [3, 0, 0, 0, 0, 0, 3],
    [0, 0, 5, 4, 0, 0, 4],
    [5, 0, 4, 0, 0, 0, 0],
])
# --------------------------------------------------------------
# Calculate cosine similarity between users
similarity_matrix = cosine_similarity(ratings)
print("User Similarity Matrix:\n", similarity_matrix)
# --------------------------------------------------------------
def recommend_items(ratings_matrix, user_index, top_n=2):
    # Get similarity scores for the target user
    user_similarity_scores = similarity_matrix[user_index]
    print(f"\nSimilarity scores for user {user_index}:", user_similarity_scores)
    # Sort similarity scores in descending order and get top-n similar user indices
    similar_users = np.argsort(user_similarity_scores)[::-1]
    # Ensure the target user itself is excluded
    similar_users = similar_users[similar_users != user_index][:top_n]
    print(f"Top-{top_n} similar users for user {user_index}:", similar_users)
    # --------------------------------------------------------------
    # Initialize dictionary to store predicted ratings for items
    recommendations = {}
    # Get the current user's ratings
    user_ratings = ratings_matrix[user_index]
    # --------------------------------------------------------------
    # Calculate weighted average ratings for each unrated item
    for item_idx in range(ratings_matrix.shape[1]):
        # Process only unrated items
        if user_ratings[item_idx] == 0:
            weighted_sum = 0.0
            similarity_sum = 0.0       
            # --------------------------------------------------------------
            # Iterate over similar users
            for other_user_idx in similar_users:
                other_user_rating = ratings_matrix[other_user_idx, item_idx]
                
                # Consider only non-zero ratings from similar users
                if other_user_rating > 0:
                    weighted_sum += user_similarity_scores[other_user_idx] * other_user_rating
                    similarity_sum += user_similarity_scores[other_user_idx]
            
            # --------------------------------------------------------------
            # Avoid division by zero
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                # Store the predicted rating in recommendations dictionary
                recommendations[item_idx] = predicted_rating
    # --------------------------------------------------------------
    # Sort recommended items based on predicted ratings in descending order
    recommended_items = sorted(recommendations, key=recommendations.get, reverse=True)  
    return recommended_items
# --------------------------------------------------------------
# Example usage: Recommend items for user index 0
user_index = 0
recommended_items = recommend_items(ratings, user_index)
print(f"\nRecommended items for user {user_index}: {recommended_items}")
