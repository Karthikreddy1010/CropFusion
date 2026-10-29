"""
main.py - Root entrypoint wrapper for Paper 3 pipeline.
Delegates execution to code/main.py.
"""
import os
import sys

# Add code directory to sys.path
CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from main import main

if __name__ == "__main__":
    main()
