import numpy as np

def cosine_similarity(a, b):
    """
    计算余弦相似度
    cosine_similarity = (a·b) / (||a|| * ||b||)
    返回值范围：[-1, 1]
    """
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0  # 避免除以0
    return dot_product / (norm_a * norm_b)

def cosine_distance(a, b):
    """
    余弦距离 = 1 - 余弦相似度
    """
    return 1 - cosine_similarity(a, b)

def euclidean_distance(a, b):
    """
    欧氏距离
    """
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def manhattan_distance(a, b):
    """
    曼哈顿距离（L1距离）
    """
    a = np.array(a)
    b = np.array(b)
    return np.sum(np.abs(a - b))

def correlation_coefficient(a, b):
    """
    皮尔逊相关系数
    返回值范围[-1, 1]
    """
    a = np.array(a)
    b = np.array(b)
    if a.size == 0 or b.size == 0:
        return 0
    if a.shape != b.shape:
        raise ValueError("向量a和b必须维度一致")
    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    numerator = np.sum(a_mean * b_mean)
    denominator = np.sqrt(np.sum(a_mean**2) * np.sum(b_mean**2))
    if denominator == 0:
        return 0
    return numerator / denominator

def jaccard_similarity(a, b):
    """
    Jaccard相似度，适用于集合或二元向量
    a,b应是0/1向量或者集合
    """
    a = np.array(a)
    b = np.array(b)
    if not ((set(np.unique(a)) <= {0,1}) and (set(np.unique(b)) <= {0,1})):
        raise ValueError("Jaccard相似度只适用于二元向量")
    intersection = np.sum(np.logical_and(a, b))
    union = np.sum(np.logical_or(a, b))
    if union == 0:
        return 0
    return intersection / union

def dice_coefficient(a, b):
    """
    Dice系数，类似Jaccard
    """
    a = np.array(a)
    b = np.array(b)
    if not ((set(np.unique(a)) <= {0,1}) and (set(np.unique(b)) <= {0,1})):
        raise ValueError("Dice系数只适用于二元向量")
    intersection = np.sum(np.logical_and(a, b))
    size_a = np.sum(a)
    size_b = np.sum(b)
    if size_a + size_b == 0:
        return 0
    return 2 * intersection / (size_a + size_b)

# 示例使用
if __name__ == "__main__":
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    v3 = [0, 1, 0, 1]  # 二元向量示例
    v4 = [1, 0, 1, 0]

    print("余弦相似度:", cosine_similarity(v1, v2))
    print("余弦距离:", cosine_distance(v1, v2))
    print("欧氏距离:", euclidean_distance(v1, v2))
    print("曼哈顿距离:", manhattan_distance(v1, v2))
    print("皮尔逊相关系数:", correlation_coefficient(v1, v2))
    print("Jaccard相似度:", jaccard_similarity(v3, v4))
    print("Dice系数:", dice_coefficient(v3, v4))
