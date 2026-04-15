#!/bin/bash

echo "=========================================="
echo "DRAgent Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv dragent_env
    source dragent_env/bin/activate
    echo "✓ Virtual environment created and activated"
    echo ""
fi

# Install dependencies and package (editable)
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -e .
echo "✓ Dependencies and dragent package installed"
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠ OPENAI_API_KEY not found in environment"
    echo ""
    read -p "Enter your OpenAI API key (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        export OPENAI_API_KEY="$api_key"
        echo "export OPENAI_API_KEY='$api_key'" >> ~/.bashrc
        echo "✓ API key set and saved to ~/.bashrc"
        echo ""
    else
        echo "⚠ Skipping API key setup"
        echo "  You'll need to set OPENAI_API_KEY before running the agent"
        echo "  Example: export OPENAI_API_KEY='your-key-here'"
        echo ""
    fi
else
    echo "✓ OPENAI_API_KEY found in environment"
    echo ""
fi

# Create results directory
mkdir -p results
echo "✓ Created results directory"
echo ""

# Run a quick test
echo "=========================================="
echo "Running quick test..."
echo "=========================================="
echo ""

if [ ! -z "$OPENAI_API_KEY" ]; then
    python3 -c "
from dragent import fetch_sdge_prices, fetch_caiso_carbon
print('Testing data fetching...')
prices = fetch_sdge_prices.invoke({})
print('✓ SDG&E prices fetched')
carbon = fetch_caiso_carbon.invoke({})
print('✓ CAISO carbon data fetched')
print('\\nAll systems operational!')
"
else
    echo "⚠ Skipping test (no API key)"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run demo:       python3 examples/demo.py"
echo "  2. Run evaluation: python3 scripts/evaluation.py"
echo "  3. Gradio UI:      python3 apps/app.py"
echo "  4. Interactive:    jupyter notebook notebooks/dragent_interactive.ipynb"
echo ""
echo "For help, see README.md"
echo ""
