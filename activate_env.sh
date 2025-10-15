#!/bin/bash
# LMOL Virtual Environment Activation Script
# Usage: source activate_env.sh

echo "🚀 Activating LMOL virtual environment..."
cd /root/LMOL
source lmol_env/bin/activate

echo "✅ Virtual environment activated!"
echo "📍 Current directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "📦 Virtual environment: $(which python)"

# Set PYTHONPATH for the project
export PYTHONPATH="/root/LMOL:${PYTHONPATH:-}"
echo "🔧 PYTHONPATH set to: $PYTHONPATH"

echo ""
echo "🎯 Ready to work with LMOL project!"
echo "   - Training: python scripts/train.py"
echo "   - Evaluation: python scripts/evaluate.py"
echo "   - Data generation: python scripts/generate_data.py"
echo ""
