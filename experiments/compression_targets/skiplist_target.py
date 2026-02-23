import random
from typing import Optional, List

class Node:
    """A single node in the skip list with forward pointers at each level."""
    def __init__(self, key: int, level: int):
        self.key = key
        self.forward: List[Optional['Node']] = [None] * (level + 1)

class SkipList:
    """
    A probabilistic data structure that allows O(log n) average-case
    search, insertion, and deletion within an ordered sequence.
    
    Uses randomized tower heights to maintain balance without
    explicit rebalancing like AVL or Red-Black trees.
    """
    def __init__(self, max_level: int = 16, p: float = 0.5):
        self.max_level = max_level
        self.probability = p
        self.header = Node(-1, self.max_level)
        self.level = 0
        self._size = 0

    @property
    def size(self) -> int:
        """Return the number of elements in the skip list."""
        return self._size

    def _random_level(self) -> int:
        """Generate a random level for a new node using geometric distribution."""
        lvl = 0
        while random.random() < self.probability and lvl < self.max_level:
            lvl += 1
        return lvl

    def insert(self, key: int) -> bool:
        """
        Insert a key into the skip list.
        Returns True if the key was inserted, False if it already existed.
        """
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if current is not None and current.key == key:
            return False  # Key already exists
        rlevel = self._random_level()
        if rlevel > self.level:
            for i in range(self.level + 1, rlevel + 1):
                update[i] = self.header
            self.level = rlevel
        new_node = Node(key, rlevel)
        for i in range(rlevel + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
        self._size += 1
        return True

    def search(self, key: int) -> bool:
        """Search for a key in the skip list. Returns True if found."""
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        current = current.forward[0]
        return current is not None and current.key == key

    def delete(self, key: int) -> bool:
        """
        Delete a key from the skip list.
        Returns True if the key was found and deleted, False otherwise.
        """
        update = [None] * (self.max_level + 1)
        current = self.header
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if current is None or current.key != key:
            return False
        for i in range(self.level + 1):
            if update[i].forward[i] != current:
                break
            update[i].forward[i] = current.forward[i]
        while self.level > 0 and self.header.forward[self.level] is None:
            self.level -= 1
        self._size -= 1
        return True

    def to_sorted_list(self) -> List[int]:
        """Return all elements as a sorted list by traversing level 0."""
        result = []
        node = self.header.forward[0]
        while node:
            result.append(node.key)
            node = node.forward[0]
        return result

    def display(self) -> None:
        """Print each level of the skip list for debugging."""
        print(f"Skip List (size={self._size}, levels={self.level + 1}):")
        for lvl in range(self.level, -1, -1):
            nodes = []
            node = self.header.forward[lvl]
            while node:
                nodes.append(str(node.key))
                node = node.forward[lvl]
            print(f"  L{lvl}: {' -> '.join(nodes)}")

if __name__ == "__main__":
    sl = SkipList(max_level=4, p=0.5)
    test_data = [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]
    for val in test_data:
        sl.insert(val)
    sl.display()
    print(f"\nSearch 19: {sl.search(19)}")
    print(f"Size: {sl.size}")
    sl.delete(19)
    print("\nAfter deleting 19:")
    sl.display()
    print(f"Sorted: {sl.to_sorted_list()}")
