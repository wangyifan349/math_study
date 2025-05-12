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






