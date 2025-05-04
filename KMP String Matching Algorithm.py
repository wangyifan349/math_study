# KMP String Matching Algorithm
def kmp_search(text, pattern):
    # Compute the partial match table (LPS array)
    def compute_lps(pattern):
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

    lps = compute_lps(pattern)
    i = 0  # Index for text
    j = 0  # Index for pattern
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            print(f"Pattern found at index {i - j}")
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
# --------------------------------------------
# Bubble Sort Algorithm
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]  # Swap elements
                swapped = True
        if not swapped:
            break
    print("Sorted array is:", arr)
# --------------------------------------------
# Binary Search Algorithm
def binary_search(arr, x):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == x:
            print(f"Element {x} is present at index {mid}")
            return mid
        elif arr[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    print(f"Element {x} is not present in array")
    return -1
# --------------------------------------------
# Example usage of KMP
text = "abcxabcdabcdabcy"
pattern = "abcdabcy"
kmp_search(text, pattern)  # Should output: "Pattern found at index 8"
# Example usage of Bubble Sort
arr = [64, 34, 25, 12, 22, 11, 90]
bubble_sort(arr)  # Should output: "Sorted array is: [11, 12, 22, 25, 34, 64, 90]"
# Example usage of Binary Search
arr = [2, 3, 4, 10, 40]
x = 10
binary_search(arr, x)  # Should output: "Element 10 is present at index 3"
