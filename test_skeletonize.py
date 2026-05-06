#!/usr/bin/env python3
"""Minimal test for skeletonize import."""

import sys

try:
    from skimage.morphology import skeletonize
    print(f"✓ Successfully imported skeletonize")
    print(f"  Module path: {skeletonize.__module__}")
    print(f"  Function: {skeletonize}")
except ImportError as e:
    print(f"✗ Failed to import skeletonize: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

print("SUCCESS")
