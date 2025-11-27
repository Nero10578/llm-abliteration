#!/bin/bash

# Script to analyze GPT-OSS-20B model structure
# Run this to understand the parameter dimensions before fixing ablation

echo "=== GPT-OSS-20B Model Structure Analysis ==="
echo ""

# Basic structure analysis
echo "1. Basic structure analysis..."
python analyze_model_structure.py --model /home/arli/models/gpt-oss-20b-BF16/ --detailed

echo ""
echo "2. Layer 11 detailed analysis (where error occurred)..."
python analyze_model_structure.py --model /home/arli/models/gpt-oss-20b-BF16/ --layer 11 --detailed

echo ""
echo "3. Layer 15 analysis (best measurement layer)..."
python analyze_model_structure.py --model /home/arli/models/gpt-oss-20b-BF16/ --layer 15 --detailed

echo ""
echo "=== Analysis Complete ==="
echo "Review the output above to understand parameter structures and fix ablation logic."