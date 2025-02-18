# Phi-4-Local
## Description
Run Microsoft's Phi-4 model locally... Yup...

## Setup Instructions

```
# Create a virtual environment named "phi-env"
python -m venv phi-env

# Activate it (Linux/macOS)
source phi-env/bin/activate

# On Windows
phi-env\Scripts\activate

# Install requirements
pip install requirements.txt

# Run the chat application
python phi_chat.py
```

After you run the script, the model will be pulled, and then the chat interface will be available at `http://127.0.0.1:7860`
