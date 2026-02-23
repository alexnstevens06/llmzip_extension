"""
LLMzip Qwen Wrapper
-------------------
This script is now a thin wrapper around and core.llmzip_utils.LLMzip to maintain
backward compatibility with existing experiments.

Deprecated: Please use core.llmzip_utils directly for new work.
"""

import os
import sys

# Add the project root to sys.path if not already present to allow core imports
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from core.llmzip_utils import LLMzip
except ImportError:
    # Handle direct execution or differing PYTHONPATH
    sys.path.append(os.path.join(project_root, "core"))
    from llmzip_utils import LLMzip


class LLMzipQwen(LLMzip):
    """
    Deprecated: Wrapper for LLMzip. Use core.llmzip_utils.LLMzip instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No additional Qwen-specific logic needed as LLMzip is robust
        pass


if __name__ == "__main__":
    print("This file is now a utility wrapper. Please use core.llmzip_utils for direct integration.")
