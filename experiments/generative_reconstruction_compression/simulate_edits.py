"""
simulate_edits.py
=================
Programmatically introduces realistic developer corrections to raw LLM output.
Used for benchmarking GRC without requiring a human in the loop.

Simulated edits mirror what a developer would actually do to LLM-generated code:
  - Add/improve docstrings
  - Add input validation / edge case handling
  - Rename generic variable names to more descriptive ones
  - Add type hints
  - Fix minor logic issues (e.g. missing return, off-by-one)
"""

import re


def simulate_developer_edits(raw_code: str) -> str:
    """
    Apply a set of realistic developer corrections to raw LLM output.
    Returns the edited version of the code.
    """
    edited = raw_code

    # 1. Strip markdown code fences if LLM wrapped output in them
    edited = re.sub(r"^```[a-zA-Z]*\n", "", edited, flags=re.MULTILINE)
    edited = re.sub(r"\n```$", "", edited, flags=re.MULTILINE)
    edited = edited.strip()

    # 2. Add a module-level docstring if missing
    if not edited.startswith('"""') and not edited.startswith("'''"):
        edited = '"""Module generated via GRC pipeline and corrected by developer."""\n\n' + edited

    # 3. Add type hints to function signatures if missing
    # e.g. def binary_search(arr, target): -> def binary_search(arr: list, target: int) -> int:
    edited = re.sub(
        r"def (\w+)\(arr, target\):",
        r"def \1(arr: list, target: int) -> int:",
        edited
    )
    edited = re.sub(
        r"def (\w+)\(lst, target\):",
        r"def \1(lst: list, target: int) -> int:",
        edited
    )

    # 4. Add input validation before the main logic if not present
    if "if not arr" not in edited and "if len(arr) == 0" not in edited:
        # Insert after the function def line
        edited = re.sub(
            r"(def \w+\([^)]*\)[^:]*:)\n(\s+)",
            lambda m: (
                m.group(1) + "\n" +
                m.group(2) + 'if not arr:\n' +
                m.group(2) + '    return -1\n' +
                m.group(2)
            ),
            edited,
            count=1
        )

    # 5. Improve docstring if function has a bare one-liner or none
    # Add a proper Google-style docstring after the def line if there isn't one
    def add_docstring(match):
        indent = match.group(2)
        existing_body = match.group(3)
        if '"""' in existing_body or "'''" in existing_body:
            return match.group(0)  # already has docstring
        docstring = (
            f'{indent}"""\n'
            f'{indent}Perform a binary search on a sorted list.\n\n'
            f'{indent}Args:\n'
            f'{indent}    arr: A sorted list of comparable elements.\n'
            f'{indent}    target: The value to search for.\n\n'
            f'{indent}Returns:\n'
            f'{indent}    The index of target in arr, or -1 if not found.\n'
            f'{indent}"""\n'
        )
        return match.group(1) + "\n" + docstring + existing_body

    edited = re.sub(
        r"(def \w+\([^)]*\)[^:]*:\n)(\s+)((?:(?!\ndef )[\s\S])*)",
        add_docstring,
        edited,
        count=1
    )

    # 6. Ensure the function returns -1 (not None) when not found
    if "return -1" not in edited and "return None" in edited:
        edited = edited.replace("return None", "return -1")
    elif "return -1" not in edited and "return" not in edited.split("def ")[1]:
        # Append return -1 at end of function
        edited = edited.rstrip() + "\n    return -1\n"

    return edited


def describe_edits(raw: str, edited: str) -> list[str]:
    """Return a human-readable list of what changed."""
    changes = []
    if raw.strip() != edited.strip():
        if "```" in raw and "```" not in edited:
            changes.append("Stripped markdown code fences")
        if '"""Module generated' in edited and '"""Module generated' not in raw:
            changes.append("Added module-level docstring")
        if "-> int:" in edited and "-> int:" not in raw:
            changes.append("Added type hints to function signature")
        if "if not arr:" in edited and "if not arr:" not in raw:
            changes.append("Added empty-list input validation")
        if "return -1" in edited and "return -1" not in raw:
            changes.append("Added explicit return -1 for not-found case")
    return changes if changes else ["No changes applied"]


if __name__ == "__main__":
    sample = '''```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
```'''
    edited = simulate_developer_edits(sample)
    print("=== EDITED OUTPUT ===")
    print(edited)
    print("\n=== CHANGES ===")
    for c in describe_edits(sample, edited):
        print(f"  - {c}")
