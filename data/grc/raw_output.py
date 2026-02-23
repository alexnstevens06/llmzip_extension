Here's a clean, iterative implementation of binary search in Python:

```python
def binary_search(arr, target):
    """
    Perform binary search on a sorted list.
    
    Args:
        arr: A sorted list of comparable elements
        target: The value to search for
        
    Returns:
        int: Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_val = arr[mid]
        
        if mid_val == target:
            return mid
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1


# Example usage
if __name__ == "__main__":
    sorted_list = [1, 3, 5, 7, 9, 11, 13, 15]
    
    print(binary_search(sorted_list, 7))   # Output: 3
    print(binary_search(sorted_list, 4))   # Output: -1
    print(binary_search([], 5))            # Output: -1
```

**Key details:**
- **Time Complexity:** O(log n) — halves the search space each iteration
- **Space Complexity:** O(1) — only uses a few variables
- **Precondition:** The input list must be sorted (ascending or descending; this assumes ascending)

**Alternative: Recursive version**
```python
def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
        
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

**Note:** For very large lists (billions of elements), use `mid = left + (right - left) // 2` instead of `(left + right) // 2` to avoid potential integer overflow (though this is rare in Python due to arbitrary precision integers).