def bubble_sort(arr):
    """
    冒泡排序：通过依次比较相邻元素并交换，逐步将最大的元素移至最后。
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                # 交换相邻元素
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果没有进行交换，提前退出
        if not swapped:
            break
    return arr
# ------------------------------------------
def selection_sort(arr):
    """
    选择排序：在未排序序列中找到最小（大）元素放到已排序序列的末尾。
    """
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        # 交换找到的最小元素到当前未排序部分的起始位置
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr
# ------------------------------------------
def insertion_sort(arr):
    """
    插入排序：将每个元素插入到已排序部分的适当位置。
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
# ------------------------------------------
def merge_sort(arr):
    """
    归并排序：采用分治法，将列表分为子列表，分别排序后合并。
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr
# ------------------------------------------
def quick_sort(arr):
    """
    快速排序：选择一个基准元素，对列表进行分区，小于基准的在左，大于基准的在右。
    """
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less = []
        greater = []
        for x in arr[1:]:
            if x <= pivot:
                less.append(x)
            else:
                greater.append(x)
        return quick_sort(less) + [pivot] + quick_sort(greater)
# ------------------------------------------
def heapify(arr, n, i):
    """
    堆化：构建一个最大堆。
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[i] < arr[left]:
        largest = left
    if right < n and arr[largest] < arr[right]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
def heap_sort(arr):
    """
    堆排序：利用堆结构，反复选择最大元素。
    """
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
# ------------------------------------------
def shell_sort(arr):
    """
    希尔排序：通过将元素间隔地分成多组，对每组进行插入排序。
    """
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr
# ------------------------------------------
# 测试用例
arr = [64, 34, 25, 12, 22, 11, 90]
print("冒泡排序结果：", bubble_sort(arr.copy()))
print("选择排序结果：", selection_sort(arr.copy()))
print("插入排序结果：", insertion_sort(arr.copy()))
print("归并排序结果：", merge_sort(arr.copy()))
print("快速排序结果：", quick_sort(arr.copy()))
print("堆排序结果：", heap_sort(arr.copy()))
print("希尔排序结果：", shell_sort(arr.copy()))




def compute_lps_array(pattern):
    """
    Compute the longest prefix that is also a suffix array for KMP algorithm.
    """
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps
def kmp_search(text, pattern):
    """
    Perform KMP search algorithm to find pattern in text.
    """
    m = len(pattern)
    n = len(text)
    lps = compute_lps_array(pattern)
    i = 0  # index for text[]
    j = 0  # index for pattern[]
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == m:
            print(f"Pattern found at index {i - j}")
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
# ------------------------------------------
def levenshtein_distance(str1, str2):
    """
    Calculate the Levenshtein distance between two strings.
    """
    if len(str1) < len(str2):
        return levenshtein_distance(str2, str1)
    if len(str2) == 0:
        return len(str1)
    previous_row = range(len(str2) + 1)
    for i, char1 in enumerate(str1):
        current_row = [i + 1]
        for j, char2 in enumerate(str2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (char1 != char2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
# ------------------------------------------
import numpy as np
def cosine_similarity(vector1, vector2):
    """
    Calculate the cosine similarity between two vectors.
    """
    dot_product = np.dot(vector1, vector2)
    norm_vec1 = np.linalg.norm(vector1)
    norm_vec2 = np.linalg.norm(vector2)
    return dot_product / (norm_vec1 * norm_vec2)
# ------------------------------------------
import math
def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in Euclidean space.
    """
    distance = 0
    for p1, p2 in zip(point1, point2):
        distance += (p1 - p2) ** 2
    return math.sqrt(distance)
# ------------------------------------------
# Example usage
text = "ababcabcabababd"
pattern = "ababd"
kmp_search(text, pattern)
str1 = "kitten"
str2 = "sitting"
lev_distance = levenshtein_distance(str1, str2)
print(f"Levenshtein distance: {lev_distance}")
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
cos_similarity = cosine_similarity(vec1, vec2)
print(f"Cosine similarity: {cos_similarity}")
point1 = (1, 2, 3)
point2 = (4, 5, 6)
euc_distance = euclidean_distance(point1, point2)
print(f"Euclidean distance: {euc_distance}")



def compute_prefix_function(pattern):
    """
    创建部分匹配表（前缀函数）。
    这个函数为KMP算法计算"失配数组"。
    """
    m = len(pattern)
    prefix_function = [0] * m
    j = 0  # 长度为j的最长前缀
    # 从第二个字符开始
    for i in range(1, m):
        # 不匹配时，回退到前一个最长前缀结尾
        while j > 0 and pattern[i] != pattern[j]:
            j = prefix_function[j - 1]
            print(f"不匹配回退: j = {j}")
        # 如果匹配，增加最长前缀的长度
        if pattern[i] == pattern[j]:
            j += 1
        prefix_function[i] = j
        print(f"计算prefix_function[{i}]: {prefix_function[i]} (根据pattern中的最长前缀)")
    return prefix_function

def kmp_search(text, pattern):
    """
    在文本text中查找模式pattern出现的位置。
    输出模式开始的所有位置。
    """
    n = len(text)
    m = len(pattern)
    prefix_function = compute_prefix_function(pattern)
    j = 0  # 模式中的位置索引
    print(f"\n开始KMP搜索。文本长度: {n}，模式长度: {m}")
    for i in range(n):
        # 当字符不匹配时，使用部分匹配表进行跳跃
        while j > 0 and text[i] != pattern[j]:
            j = prefix_function[j - 1]
            print(f"字符不匹配，使用部分匹配表跳跃：j = {j}")
        # 如果找到匹配
        if text[i] == pattern[j]:
            j += 1
        # 如果整个模式匹配，输出位置
        if j == m:
            print(f"模式匹配开始位置: {i - m + 1}")
            j = prefix_function[j - 1]  # 准备寻找下一个可能的匹配
# 示例使用
text = "ababcabcabababd"
pattern = "ababd"
print("计算前缀函数（部分匹配表）：")
prefix_function = compute_prefix_function(pattern)
print(f"部分匹配表: {prefix_function}\n")
kmp_search(text, pattern)

"""
KMP算法的核心思想是通过一个有效的预处理步骤，将模式串的信息存储在一个辅助数据结构中，从而在主串中进行快速匹配。以下是KMP算法的详细逻辑分解：
- 首先，我们要构建一个称为部分匹配表或前缀函数的辅助数组。该数组用于存储模式串中每个位置的前缀匹配信息。具体来说，数组中的每个元素记录了当前字符之前的子串中，最大长度的相同前缀和后缀的长度。
- 在构建部分匹配表时，我们初始化一个与模式串长度相等的数组，所有位置的初值设为0。接下来，使用两个指针遍历模式串：指针`i`用于当前字符的索引，而指针`j`用于记录前缀匹配的长度。
- 当模式串的当前字符和前一个最大前缀的后一个字符匹配时，我们增加`j`的值，同时记录在部分匹配表中。当字符不匹配时，我们利用部分匹配表中的信息跳回前一个可能的前缀位置，并继续检查，直到`j`为0或者匹配。在`j`为0时，无法找到更小的前缀继续匹配，所以直接更新当前的前缀长度为0。
- 完成上一步的预处理后，我们开始在主串中执行搜索。我们用两个指针遍历主串和模式串，开始从`i = 0`和`j = 0`的位置，将主串的字符与模式串的当前字符进行匹配。
- 一旦字符匹配，我们将两个指针同时向右移动，继续匹配接下来的字符。如果`j`增加至与模式串长度相同，说明一次完整匹配成功。此时，记录匹配在主串中首次出现的位置，然后使用部分匹配表优化后的位置来调整模式串，以继续检查主串中可能的其他匹配。
- 如果字符不匹配，我们利用部分匹配表直接跳转到模式串中前一个最大前缀的位置，使得模式串向右移动时避免了冗余的匹配。继续这个过程直到模式串找到其完整匹配，或者主串字符已无剩余进行匹配。
- KMP算法的优点在于，它在预处理阶段对模式串进行了一次遍历构建表格，然后仅用线性时间复杂度遍历主串。通过预处理的前缀信息跳过不必要的匹配，使KMP算法可以高效地在主串中搜索模式子串。它被广泛应用于文本编辑器的搜索功能、DNA序列比对、以及其他需要频繁进行字符串匹配的场景中。


pattern = "ababd"
Index:  0  1  2  3  4
Chars:  a  b  a  b  d
Prefix: -  -  a  ab -
部分匹配表通过记录每个位置的最长公共前后缀长度，
帮助KMP算法在匹配过程中有效跳过已知可匹配的部分。
"""


