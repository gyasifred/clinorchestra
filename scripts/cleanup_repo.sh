#!/bin/bash

# ClinOrchestra Repository Cleanup Script
# Removes temporary files, cache directories, and build artifacts

set -e

echo "==================================="
echo "ClinOrchestra Repository Cleanup"
echo "==================================="
echo ""

# Navigate to repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Repository root: $REPO_ROOT"
echo ""

# Count files before cleanup
echo "Counting files before cleanup..."
BEFORE_COUNT=$(find . -type f | wc -l)
echo "Total files before: $BEFORE_COUNT"
echo ""

# 1. Remove Python cache files
echo "1. Removing Python cache files..."
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "   ✓ Python cache cleaned"

# 2. Remove temporary files
echo "2. Removing temporary files..."
find . -type f \( -name "*~" -o -name "*.tmp" -o -name "*.bak" -o -name ".DS_Store" \) -delete 2>/dev/null || true
echo "   ✓ Temporary files removed"

# 3. Remove LaTeX build artifacts (keep .tex source)
echo "3. Removing LaTeX build artifacts..."
find docs/ -type f \( -name "*.aux" -o -name "*.log" -o -name "*.nav" \
    -o -name "*.out" -o -name "*.snm" -o -name "*.toc" -o -name "*.vrb" \) -delete 2>/dev/null || true
echo "   ✓ LaTeX artifacts removed"

# 4. Remove editor swap files
echo "4. Removing editor swap files..."
find . -type f \( -name "*.swp" -o -name "*.swo" -o -name "*~" \) -delete 2>/dev/null || true
echo "   ✓ Editor swap files removed"

# 5. Remove empty directories (except .git)
echo "5. Removing empty directories..."
find . -type d -empty -not -path "./.git/*" -delete 2>/dev/null || true
echo "   ✓ Empty directories removed"

# Count files after cleanup
echo ""
echo "Counting files after cleanup..."
AFTER_COUNT=$(find . -type f | wc -l)
REMOVED=$((BEFORE_COUNT - AFTER_COUNT))
echo "Total files after: $AFTER_COUNT"
echo "Files removed: $REMOVED"
echo ""

# Summary
echo "==================================="
echo "Cleanup completed successfully!"
echo "==================================="
echo ""
echo "Recommendations:"
echo "  - Review .gitignore to prevent future cache files"
echo "  - Run 'git status' to check for untracked files"
echo "  - Consider 'git clean -fdx -n' for dry-run of git clean"
echo ""

# Optional: Show size savings
if command -v du &> /dev/null; then
    REPO_SIZE=$(du -sh . | cut -f1)
    echo "Current repository size: $REPO_SIZE"
fi

exit 0
