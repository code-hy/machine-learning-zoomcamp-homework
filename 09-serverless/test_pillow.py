#!/usr/bin/env python3
"""Test script to verify Pillow installation"""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    from PIL import Image
    print(f"✅ SUCCESS: Pillow imported successfully!")
    print(f"Pillow version: {Image.__version__}")
    print(f"PIL location: {Image.__file__}")
except ImportError as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
