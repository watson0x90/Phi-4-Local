# Phi-4-Local with RAG and Persistent History

## Description
Run Microsoft's Phi-4 model locally with enhanced features:
- **Persistent Chat History**: All conversations are saved and can be loaded later
- **Document Knowledge Base (RAG)**: Upload PDFs, Word docs, and text files to create a searchable knowledge base
- **Intelligent Context Retrieval**: The model can reference uploaded documents to provide more accurate answers

## Features

### 1. Persistent Chat History
- Automatically saves all conversations with timestamps
- Create new chat sessions
- Load previous sessions from a dropdown menu
- Chat history stored in `chat_history.json`

### 2. Document Knowledge Base (RAG)
- Support for multiple file formats: PDF, DOCX, TXT, MD
- Documents are chunked and indexed for efficient retrieval
- Uses semantic search to find relevant information
- Index stored locally for persistence across sessions

### 3. Enhanced Chat Interface
- Clean Gradio interface with separate panels
- RAG settings to control document retrieval
- Real-time document upload status
- Session management controls

## Setup Instructions

```bash
# Create a virtual environment named "phi-env"
python -m venv phi-env

# Activate it (Linux/macOS)
source phi-env/bin/activate

# On Windows
phi-env\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the enhanced chat application
python phi_chat.py
```

## Usage Guide

### Starting a Chat
1. Run the application: `python phi_chat.py`
2. Open your browser to `http://127.0.0.1:7860`
3. Start typing in the message box - a new session is created automatically

### Uploading Documents
1. Click on "Choose File" in the Document Upload section
2. Select a PDF, DOCX, TXT, or MD file
3. Click "Add to Knowledge Base"
4. The document will be processed and indexed automatically

### Using the Knowledge Base
- Keep "Use Knowledge Base" checked to enable RAG
- Adjust "Number of relevant documents" to control how much context is retrieved
- The model will automatically reference relevant documents when answering

### Managing Chat Sessions
- **New Session**: Start a fresh conversation
- **Load Session**: Select a previous session from the dropdown
- **Clear Chat**: Clear the current display (doesn't delete history)
- **Refresh Sessions**: Update the session list

## File Structure
```
phi-4-local/
├── phi_chat.py    # Main application with RAG support
├── requirements.txt        # Python dependencies
├── chat_history.json      # Saved chat sessions (created automatically)
├── document_index/        # Document embeddings and index (created automatically)
│   ├── documents.json     # Document metadata
│   └── faiss.index       # Vector embeddings index
└── README.md             # This file
```

## Tips for Best Results

1. **Document Quality**: Upload clean, well-formatted documents for best results
2. **Specific Questions**: Ask specific questions about uploaded documents for targeted answers
3. **Context Window**: The model considers recent chat history plus relevant documents
4. **Multiple Documents**: You can upload multiple documents to build a comprehensive knowledge base

## Troubleshooting

### Out of Memory Errors
- Reduce the number of retrieved documents (rag_k parameter)
- Consider using smaller chunks for documents
- Ensure you have sufficient RAM for the Phi-4 model

### Slow Performance
- First model load takes time as it downloads from Hugging Face
- Document indexing may take time for large PDFs
- Consider using GPU acceleration if available

### Document Processing Issues
- Ensure documents are not password-protected
- Check file encoding for text files (UTF-8 recommended)
- Some complex PDF layouts may not extract perfectly

## Next Steps
- Fine-tune the model for specific domains
- Add support for more document formats (Excel, HTML, etc.)
- Implement document management UI (delete, update, view)
- Add conversation export features
- Implement multi-user support
