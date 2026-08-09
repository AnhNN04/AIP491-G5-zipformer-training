# Setup environment for VietASR
# Usage: source setup.sh

# Detect if the script is being sourced or executed directly
if [ "$$" -eq "$BASHPID" ] || [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    echo "Warning: You are running this script directly. To apply changes to your current shell,"
    echo "please run: source setup.sh"
    echo ""
fi

# 1. Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment (.venv) activated."
else
    echo "Warning: .venv/bin/activate not found. Please make sure the virtual environment exists."
fi

# Get the absolute root path of the project
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 2. Export necessary paths to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/ASR/zipformer:${PYTHONPATH:-}"
echo "PYTHONPATH configured with project directories."

# 3. Export protobuf environment variable to prevent segfaults
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
echo "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION set to python."

echo "Environment setup complete! Ready for testing and inference."
