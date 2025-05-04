import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example user-item rating matrix
ratings = np.array([
    [4, 0, 0, 5, 1, 0, 0],
    [5, 5, 4, 0, 0, 0, 0],
    [0, 0, 0, 2, 4, 5, 0],
    [3, 0, 0, 0, 0, 0, 3],
    [0, 0, 5, 4, 0, 0, 4],
    [5, 0, 4, 0, 0, 0, 0],
])

# Transpose the rating matrix to align items as rows
# Rows are now items, columns are users
ratings_T = ratings.T

# --------------------------------------------------------------
# Calculate cosine similarity between items
item_similarity_matrix = cosine_similarity(ratings_T)
print("Item Similarity Matrix:\n", item_similarity_matrix)

# --------------------------------------------------------------
def recommend_items_based_on_items(ratings_matrix, user_index, top_n=2):
    # Get user's ratings
    user_ratings = ratings_matrix[user_index]
    recommendations = {}
    
    # --------------------------------------------------------------
    # Calculate scores for each item not rated by the user
    for item_idx in range(ratings_matrix.shape[1]):
        if user_ratings[item_idx] == 0:  # Only consider unrated items
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            # --------------------------------------------------------------
            # Iterate over all items to calculate similarity
            for other_item_idx in range(ratings_matrix.shape[1]):
                if user_ratings[other_item_idx] > 0:  # Only consider rated items
                    item_similarity = item_similarity_matrix[item_idx, other_item_idx]
                    weighted_sum += item_similarity * user_ratings[other_item_idx]
                    similarity_sum += item_similarity
            
            # --------------------------------------------------------------
            # Avoid division by zero and calculate predicted rating
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations[item_idx] = predicted_rating
    
    # --------------------------------------------------------------
    # Sort recommended items based on predicted ratings
    recommended_items = sorted(recommendations, key=recommendations.get, reverse=True)
    
    return recommended_items

# --------------------------------------------------------------
# Example usage: Recommend items for user index 0
user_index = 0
recommended_items = recommend_items_based_on_items(ratings, user_index)
print(f"\nRecommended items for user {user_index}: {recommended_items}")
