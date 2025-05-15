import numpy as np

def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度：
    sim = (A · B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec1, vec2)
    normA = np.linalg.norm(vec1)
    normB = np.linalg.norm(vec2)
    if normA == 0 or normB == 0:
        # 避免除以零错误，未评分或全0向量相似度为0
        return 0.0
    return dot_product / (normA * normB)

def get_user_similarities(ratings):
    """
    计算用户与用户之间的相似度矩阵
    输入：
        ratings - numpy数组，形状 (用户数, 物品数)
    输出：
        user_sim_matrix - numpy数组，形状 (用户数, 用户数)
    """
    n_users = ratings.shape[0]
    user_sim_matrix = np.zeros((n_users, n_users))

    for i in range(n_users):
        for j in range(i, n_users):
            sim = cosine_similarity(ratings[i], ratings[j])
            user_sim_matrix[i, j] = sim
            user_sim_matrix[j, i] = sim  # 对称矩阵
    return user_sim_matrix

def get_item_similarities(ratings):
    """
    计算物品与物品之间的相似度矩阵
    输入：
        ratings - numpy数组，形状 (用户数, 物品数)
    输出：
        item_sim_matrix - numpy数组，形状 (物品数, 物品数)
    """
    n_items = ratings.shape[1]
    item_sim_matrix = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(i, n_items):
            sim = cosine_similarity(ratings[:, i], ratings[:, j])
            item_sim_matrix[i, j] = sim
            item_sim_matrix[j, i] = sim  # 对称矩阵
    return item_sim_matrix

def predict_user_based(ratings, user_sim_matrix, user_index, k=2):
    """
    基于用户的协同过滤预测函数
    参数：
        ratings         - 评分矩阵 (用户数, 物品数)
        user_sim_matrix - 用户相似度矩阵 (用户数, 用户数)
        user_index      - 待预测的用户索引
        k               - 使用k个最相似用户加权预测
    返回：
        predicted_ratings - 预测评分向量，长度与物品数相同
    """
    n_users, n_items = ratings.shape
    predicted_ratings = np.zeros(n_items)
    sim_scores = user_sim_matrix[user_index]

    for item in range(n_items):
        if ratings[user_index, item] == 0:  # 目标用户未评分，需要预测
            # 找出对该物品评分且与目标用户最相似的k个用户
            sim_users = []
            for other_user in range(n_users):
                if other_user != user_index and ratings[other_user, item] > 0:
                    sim_users.append((other_user, sim_scores[other_user]))
            # 按相似度排序
            sim_users.sort(key=lambda x: x[1], reverse=True)
            top_k_users = sim_users[:k]

            numerator = 0.0
            denominator = 0.0
            for (other_user, sim) in top_k_users:
                numerator += sim * ratings[other_user, item]
                denominator += abs(sim)
            if denominator > 0:
                predicted_ratings[item] = numerator / denominator
            else:
                predicted_ratings[item] = 0
        else:
            # 已评分则用已有评分
            predicted_ratings[item] = ratings[user_index, item]
    return predicted_ratings

def predict_item_based(ratings, item_sim_matrix, user_index, k=2):
    """
    基于物品的协同过滤预测函数
    参数：
        ratings         - 评分矩阵 (用户数, 物品数)
        item_sim_matrix - 物品相似度矩阵 (物品数, 物品数)
        user_index      - 待预测的用户索引
        k               - 使用k个最相似物品加权预测
    返回：
        predicted_ratings - 预测评分向量，长度与物品数相同
    """
    n_users, n_items = ratings.shape
    user_ratings = ratings[user_index]
    predicted_ratings = np.copy(user_ratings)

    for item in range(n_items):
        if user_ratings[item] == 0:  # 待预测的物品
            # 找该物品与其他物品的相似度，并筛选该用户已经评分的物品
            sim_items = []
            for other_item in range(n_items):
                if other_item != item and user_ratings[other_item] > 0:
                    sim = item_sim_matrix[item, other_item]
                    sim_items.append((other_item, sim))
            # 按相似度降序排序，取前k
            sim_items.sort(key=lambda x: x[1], reverse=True)
            top_k_items = sim_items[:k]

            numerator = 0.0
            denominator = 0.0
            for (other_item, sim) in top_k_items:
                numerator += sim * user_ratings[other_item]
                denominator += abs(sim)
            if denominator > 0:
                predicted_ratings[item] = numerator / denominator
            else:
                predicted_ratings[item] = 0
        # 用户已有评分 保持不变
    return predicted_ratings

def print_ratings(title, ratings_vector, item_names):
    print(title)
    for name, rating in zip(item_names, ratings_vector):
        print(f"  {name}: {rating:.2f}")
    print('-' * 40)

if __name__ == "__main__":
    # 用户物品评分矩阵示例（行为用户，列为物品）
    # 0表示未评分
    ratings = np.array([
        [5, 3, 0, 1, 0],
        [4, 0, 0, 1, 0],
        [1, 1, 0, 5, 4],
        [0, 0, 5, 4, 5],
        [0, 1, 5, 4, 0],
    ])

    user_names = ['用户1', '用户2', '用户3', '用户4', '用户5']
    item_names = ['物品A', '物品B', '物品C', '物品D', '物品E']

    print("评分矩阵:")
    for user, user_ratings in zip(user_names, ratings):
        print(f"  {user}: {user_ratings}")
    print('='*40)

    # 计算相似度矩阵
    user_sim_matrix = get_user_similarities(ratings)
    item_sim_matrix = get_item_similarities(ratings)

    print("用户相似度矩阵:")
    print(user_sim_matrix)
    print('-' * 40)

    print("物品相似度矩阵:")
    print(item_sim_matrix)
    print('-' * 40)

    # 选择目标用户做预测，比如用户1（索引0）
    target_user_index = 0
    print(f"为用户: {user_names[target_user_index]} 预测评分（基于用户）")
    user_based_predictions = predict_user_based(ratings, user_sim_matrix, target_user_index, k=2)
    print_ratings("基于用户的预测评分结果:", user_based_predictions, item_names)

    print(f"为用户: {user_names[target_user_index]} 预测评分（基于物品）")
    item_based_predictions = predict_item_based(ratings, item_sim_matrix, target_user_index, k=2)
    print_ratings("基于物品的预测评分结果:", item_based_predictions, item_names)
