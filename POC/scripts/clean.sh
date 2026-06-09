#!/usr/bin/env bash

# Get the script directory to resolve paths relatively
SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Cleaning Python and C++ caches in $POC_ROOT..."

# Remove python caches
find "$POC_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find "$POC_ROOT" -type f -name "*.pyc" -delete 2>/dev/null
find "$POC_ROOT" -type f -name "*.pyo" -delete 2>/dev/null
find "$POC_ROOT" -type f -name "*.pyd" -delete 2>/dev/null

# Remove pytest and mypy caches
find "$POC_ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find "$POC_ROOT" -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null

# Remove k2 compiler C++ caches (specifically look for .k2_cache)
find "$POC_ROOT" -type d -name ".k2_cache" -exec rm -rf {} + 2>/dev/null

# Remove log files
find "$POC_ROOT" -type f -name "*.log" -delete 2>/dev/null

echo "Clean completed successfully!"
