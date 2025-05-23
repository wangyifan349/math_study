import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers

# ------------------------  基础概念部分  ------------------------
def cosine_similarity_func(a, b):
    """
    计算余弦相似度
    衡量两个非零向量之间夹角的余弦值。用于文本相似度计算。
    公式: cos(θ) = (A · B) / (||A|| * ||B||)，其中 A 和 B 是向量，||A|| 是 A 的欧几里得范数。
    """
    dot_product = np.dot(a, b)  # 计算向量点积
    magnitude_a = np.linalg.norm(a)  # 计算 A 的欧几里得范数
    magnitude_b = np.linalg.norm(b)  # 计算 B 的欧几里得范数
    return dot_product / (magnitude_a * magnitude_b) # 返回余弦相似度

def euclidean_distance(a, b):
    """
    计算欧几里得距离（L2 距离）
    两点之间的直线距离。
    公式: sqrt(Σ(xi - yi)^2)，其中 x 和 y 是两个向量。
    """
    return np.linalg.norm(a - b)  # 计算 L2 范数，即欧几里得距离

def bubble_sort(arr):
    """
    冒泡排序
    一种简单的排序算法，重复地遍历要排序的列表，比较相邻的元素并交换它们，直到列表排序完成。
    效率较低，但在教学中易于理解。
    """
    n = len(arr)
    for i in range(n): # 遍历所有元素
        for j in range(0, n - i - 1):  # 每次遍历都减少一次比较
            if arr[j] > arr[j + 1]:  # 如果前一个元素大于后一个元素
                arr[j], arr[j + 1] = arr[j + 1], arr[j]  # 交换元素

# ------------------------  机器学习算法部分  ------------------------
def linear_regression_example():
    """线性回归示例"""
    X = np.array([[1], [2], [3], [4], [5]])  # 特征
    y = np.array([2, 4, 5, 4, 5])  # 目标变量

    model = LinearRegression()  # 创建线性回归模型
    model.fit(X, y)  # 训练模型

    # 预测
    x_new = np.array([[6]])
    y_pred = model.predict(x_new)
    print("线性回归预测:", y_pred)

def logistic_regression_example():
    """逻辑回归示例"""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])  # 二元标签

    model = LogisticRegression() # 创建逻辑回归模型
    model.fit(X, y)  # 训练模型

    # 预测
    x_new = np.array([[3.5]])
    y_pred = model.predict(x_new)
    print("逻辑回归预测:", y_pred)  # 输出 0 或 1
    print("逻辑回归概率:", model.predict_proba(x_new))  # 输出属于每个类别的概率

def knn_example():
    """K 近邻 (KNN) 示例"""
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])

    model = KNeighborsClassifier(n_neighbors=3)  # 创建 KNN 模型，设置邻居数为 3
    model.fit(X, y)  # 训练模型

    # 预测
    x_new = np.array([[1.5, 2]])
    y_pred = model.predict(x_new)
    print("KNN 预测:", y_pred)

def decision_tree_example():
    """决策树示例"""
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = DecisionTreeClassifier()  # 创建决策树模型
    clf = clf.fit(X, Y)  # 训练模型
    print("决策树预测:", clf.predict([[2., 2.]]))

def svm_example():
    """支持向量机 (SVM) 示例"""
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()  # 创建 SVM 模型
    clf.fit(X, y)  # 训练模型
    print("SVM 预测:", clf.predict([[2., 2.]]))

def naive_bayes_example():
    """朴素贝叶斯示例"""
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([1, 1, 1, 2, 2, 2])
    clf = GaussianNB()  # 创建高斯朴素贝叶斯模型
    clf.fit(X, Y)  # 训练模型
    print("朴素贝叶斯预测:", clf.predict([[-0.8, -1]]))

def kmeans_example():
    """K-Means 聚类示例"""
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(X)  # 创建 K-Means 模型，设置簇数为 2
    print("K-Means 标签:", kmeans.labels_)  # 打印每个样本的簇标签
    print("K-Means 中心:", kmeans.cluster_centers_)  # 打印每个簇的中心点

def random_forest_example():
    """随机森林示例"""
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = RandomForestClassifier(n_estimators=10)  # 创建随机森林模型，设置树的数量为 10
    clf = clf.fit(X, Y)  # 训练模型
    print("随机森林预测:", clf.predict([[2., 2.]]))

def gradient_boosting_example():
    """梯度提升机 (GBM) 示例"""
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
       max_depth=1, random_state=0).fit(X, Y) # 创建 GBM 模型
    print("梯度提升机预测:", clf.predict([[2., 2.]]))

def pca_example():
    """主成分分析 (PCA) 示例"""
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=1)  # 创建 PCA 模型，设置降维后的维度为 1
    pca.fit(X)  # 训练模型
    print("PCA 转换:", pca.transform(X)) # 打印降维后的数据

def autoencoder_example():
    """自编码器示例"""
    # 定义自编码器模型
    encoding_dim = 32  # 压缩后的维度

    input_img = tf.keras.Input(shape=(784,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)  # 编码层
    decoded = layers.Dense(784, activation='sigmoid')(encoded)  # 解码层

    autoencoder = tf.keras.Model(input_img, decoded)

    # 编译模型
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # 准备数据 (使用 MNIST 数据集)
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # 训练模型
    autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    # 使用自编码器进行编码和解码
    encoded_imgs = autoencoder.predict(x_test)

    print("自编码器完成")

# ------------------------  其他算法部分  ------------------------
def longest_common_subsequence(s1, s2):
    """
    最长公共子序列 (LCS)
    使用动态规划。
    找到两个序列（字符或者字符串）中最长的公共子序列。
    """
    n = len(s1)
    m = len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1): # 遍历字符串 s1
        for j in range(1, m + 1): # 遍历字符串 s2
            if s1[i-1] == s2[j-1]:  # 如果当前字符相同
                dp[i][j] = dp[i-1][j-1] + 1 # LCS 长度加 1
            else: # 否则，取上方和左方的最大值
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[n][m]

# ------------------------        以下是推荐Demo，上面是机器学习的一些基础函数  ------------------------

# ------------------------  相似度计算方法  ------------------------
def pearson_correlation(user1_ratings, user2_ratings):
    """
    计算两个用户的 Pearson 相关系数，使用 scipy。
    考虑评分倾向 (例如：有些用户倾向于给高分，有些用户倾向于给低分)。
    处理评分缺失的情况（评分都为0）时，返回0。
    """
    # 只保留共同评分的items
    rated_indices1 = np.where(user1_ratings > 0)[0]
    rated_indices2 = np.where(user2_ratings > 0)[0]

    common_items = np.intersect1d(rated_indices1, rated_indices2) #获取共同评分的items

    if len(common_items) == 0: # 没有共同评分的物品
      return 0.0

    user1_common_ratings = user1_ratings[common_items]
    user2_common_ratings = user2_ratings[common_items]

    correlation, _ = scipy.stats.pearsonr(user1_common_ratings, user2_common_ratings)
    # 如果存在nan,返回0
    if np.isnan(correlation):
      return 0.0

    return correlation

def jaccard_index(item1_users, item2_users):
    """
    计算两个物品的 Jaccard 指数。
    简单易懂，计算速度快，但忽略了用户对物品的评分差异。更适合隐式反馈数据。
    item1_users: 喜欢物品1的用户索引。
    item2_users: 物品2的用户索引。

    """
    intersection = np.intersect1d(item1_users, item2_users)
    union = np.union1d(item1_users, item2_users)

    if len(union) == 0:
      return 0.0
    return len(intersection) / len(union)

# ---------------------------------------------------------------
# 2. 基于内容的推荐函数 (与之前相同)
def content_based_recommendation(items, user_liked_items, top_n):
   # 代码省略 (与前面的代码相同)
    item_ids = items.keys()
    descriptions = items.values()
    item_ids_list = list(item_ids)

    # 1. 将物品描述转换为TF-IDF向量
    vectorizer = TfidfVectorizer()
    item_vectors = vectorizer.fit_transform(descriptions)

    # 2. 创建用户画像： 取用户喜欢物品向量的平均值
    user_profile = np.zeros(item_vectors.shape[1])
    for item_id in user_liked_items:
        index = item_ids_list.index(item_id)
        item_vector = item_vectors[index]
        user_profile += item_vector.toarray().flatten()

    if len(user_liked_items) > 0:
        user_profile /= len(user_liked_items) # 平均值

    # 3. 计算用户画像与所有物品的相似度 (余弦相似度)
    similarities = cosine_similarity(user_profile.reshape(1, -1), item_vectors).flatten()

    # 4. 排除用户已经喜欢的物品 (避免重复推荐)
    for item_id in user_liked_items:
        index = item_ids_list.index(item_id)
        similarities[index] = -1  # 将相似度设置为-1，确保不被推荐

    # 5. 推荐最相似的物品
    sorted_indices = np.argsort(similarities)[::-1] # 从大到小排列
    top_indices = sorted_indices[:top_n] #选取top N

    recommendations = []
    for i in top_indices:
        recommendation = items.keys().__getitem__(i)
        recommendations.append(recommendation) # 获取推荐的物品ID

    return recommendations

# 3. 基于物品的协同过滤函数 (修改后)
def item_based_collaborative_filtering(user_item_matrix, item_names, user_index, top_n, similarity_metric="cosine"):
    """
    基于物品的协同过滤: 根据用户已喜欢的物品，推荐相似的物品（基于其他用户的行为）
    similarity_metric: 指定使用的相似度计算方法 (cosine, pearson, jaccard)
    """
    num_items = user_item_matrix.shape[1] #物品总数

    # 1. 计算物品之间的相似度
    item_similarity_matrix = np.zeros((num_items, num_items)) #初始化矩阵
    for i in range(num_items):
        for j in range(num_items):
            if similarity_metric == "cosine": #使用余弦相似性
                item_similarity_matrix[i, j] = cosine_similarity(user_item_matrix[:, i].reshape(1, -1), user_item_matrix[:, j].reshape(1, -1))[0, 0]
            elif similarity_metric == "pearson": # 使用 Pearson 相关系数
                item_similarity_matrix[i, j] = pearson_correlation(user_item_matrix[:, i], user_item_matrix[:, j])
            elif similarity_metric == "jaccard": # 使用 Jaccard 指数
                #查找喜欢物品i的用户
                user_like_item1 = np.where(user_item_matrix[:,i] > 0)[0]
                #查找喜欢物品j的用户
                user_like_item2 = np.where(user_item_matrix[:,j] > 0)[0]
                item_similarity_matrix[i, j] = jaccard_index(user_like_item1, user_like_item2)
            else:
                raise ValueError("无效的相似度计算方法。")

    # 2. 获取目标用户评价过的物品 (包括喜欢和不喜欢)
    rated_items_indices = np.where(user_item_matrix[user_index] > 0)[0]

    # 3. 计算每个物品的推荐分数
    scores = np.zeros(len(item_names))
    for liked_item_index in rated_items_indices:
        scores += item_similarity_matrix[liked_item_index] * user_item_matrix[user_index, liked_item_index]  # 用户对一个物品的得分作为权重

    # 4. 排除用户已经评价过的物品
    for rated_item_index in rated_items_indices:
        scores[rated_item_index] = -1

    # 5. 推荐分数最高的物品
    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:top_n]

    recommendations = []
    for i in top_indices:
        recommendation = item_names[i]  # 获取推荐的物品名称
        recommendations.append(recommendation)

    return recommendations

# 4. 混合推荐函数 （简单加权平均）
def hybrid_recommendation(items, user_item_matrix, item_names, user_index, top_n,
                          content_weight=0.5, cf_weight=0.5, cf_similarity="cosine"):
    """
    混合推荐:结合基于内容的推荐和协同过滤.
    cf_similarity: 指定协同过滤中使用的相似度计算方法.
    """
    # 1. 获取用户喜欢的物品
    liked_items_indices = np.where(user_item_matrix[user_index] > 3)[0]  # 评分大于3视为喜欢
    user_liked_items = []
    for i in liked_items_indices:
        item = item_names[i]
        user_liked_items.append(item)

    # 2. 执行基于内容的推荐
    content_recommendations = content_based_recommendation(items, user_liked_items, top_n=top_n)

    # 3. 执行基于物品的协同过滤
    cf_recommendations = item_based_collaborative_filtering(user_item_matrix, item_names, user_index, top_n=top_n, similarity_metric=cf_similarity)

    # 4. 合并推荐结果
    hybrid_recommendations = content_recommendations + cf_recommendations

    # Remove duplicate items
    seen = set()
    unique_recommendations = []
    for item in hybrid_recommendations:
        if item not in seen:
            unique_recommendations.append(item)
            seen.add(item)

    final_recommendations = unique_recommendations[:top_n]

    return final_recommendations

# 5. 演示用户界面
def display_gui():
    """
    用户界面演示
    """
    print("---------------------------------------------")
    print("        ALL Machine Learning Demo")
    print("---------------------------------------------")

    # 打印函数
    print("1. 基础算法:")
    print("   1.1 余弦相似度")
    print("   1.2 欧几里得距离")
    print("   1.3 冒泡排序")
    print("2. 机器学习Demo")
    print("   2.1 线性回归")
    print("   2.2 逻辑回归")
    print("   2.3 KNN")
    print("   2.4 决策树")
    print("   2.5 svm")
    print("   2.6 朴素贝叶斯")
    print("   2.7 K-Means 聚类")
    print("   2.8 随机森林")
    print("   2.9 梯度提升机")
    print("   2.10 主成分分析")
    print("3. 推荐Demo  ")

    while True:
        print("\n请选择你想查看的demo：")
        choice = input("请输入序号：")

        if choice == '0':
            break
        elif choice == "1.1":
            vector_a = np.array([1, 2, 3])
            vector_b = np.array([4, 5, 6])
            similarity = cosine_similarity_func(vector_a, vector_b)
            print(f"余弦相似度: {similarity}")
        elif choice == "1.2":
            point_a = np.array([1, 2])
            point_b = np.array([4, 6])
            distance = euclidean_distance(point_a, point_b)
            print(f"欧几里得距离: {distance}")
        elif choice == "1.3":
            data = [64, 34, 25, 12, 22, 11, 90]
            bubble_sort(data)
            print(f"排序后的数组: {data}")
        elif choice == "2.1":
            linear_regression_example()
        elif choice == "2.2":
            logistic_regression_example()
        elif choice == "2.3":
            knn_example()
        elif choice == "2.4":
            decision_tree_example()
        elif choice == "2.5":
            svm_example()
        elif choice == "2.6":
            naive_bayes_example()
        elif choice == "2.7":
            kmeans_example()
          #  autoencoder_example() #由于mnist下载问题，不能直接运行
        elif choice == "2.8":
            random_forest_example()
        elif choice == "2.9":
            gradient_boosting_example()
        elif choice == "2.10":
            pca_example()
        elif choice == "3":
            while True:
                print("\n请选择用户：")
                for i, user_name in enumerate(user_names):
                    print(f"{i+1}. {user_name}")
                print("0. 退出")

                try:
                    choice = int(input("请输入用户编号："))

                    if choice == 0:
                        break
                    elif 1 <= choice <= len(user_names):
                        user_index = choice - 1
                        user_name = user_names[user_index]

                        # 获取混合推荐，可以指定CF的相似度计算方法
                        top_n = 3
                        cf_similarity_metric = input("请选择协同过滤的相似度计算方法 (cosine, pearson, jaccard, 默认为cosine): ") or "cosine"
                        recommendations = hybrid_recommendation(items_demo, user_item_matrix, item_names, user_index, top_n=top_n, cf_similarity=cf_similarity_metric)

                        print(f"\n为 {user_name} 推荐的物品 (使用 {cf_similarity_metric})：")
                        for recommendation in recommendations:
                            print(f"- {recommendation}: {items_demo[recommendation]}")

                    else:
                        print("无效的输入，请重新选择。")

                except ValueError:
                    print("无效的输入，请输入数字。")

        else:
            print("无效的输入，请重新选择。")

# ------------------------  运行Demo  ------------------------
# 1. 准备推荐Demo数据
items_demo = {
    'item1': 'action adventure science fiction superhero',
    'item2': 'science fiction space travel adventure',
    'item3': 'comedy romance drama',
    'item4': 'action thriller crime mystery',
    'item5': 'fantasy adventure magic',
    'item6': 'documentary nature wildlife'
}

user_item_matrix = np.array([
    [5, 4, 0, 0, 0, 0],  # 用户1
    [0, 5, 3, 0, 0, 0],  # 用户2
    [0, 0, 0, 4, 5, 0],  # 用户3
    [5, 0, 0, 0, 0, 3],  # 用户4
    [0, 0, 5, 4, 0, 0]   # 用户5
])
item_names = list(items_demo.keys())
user_names = ['User 1', 'User 2', 'User 3', 'User 4', 'User 5']

display_gui()
