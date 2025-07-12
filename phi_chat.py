# Save this file as phi_chat_enhanced.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import docx
from pathlib import Path

# Specify the model name
model_name = "microsoft/phi-4"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load embedding model for RAG
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

class DocumentStore:
    def __init__(self, index_path="document_index"):
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_index()
    
    def load_index(self):
        """Load existing index if available"""
        docs_file = self.index_path / "documents.json"
        index_file = self.index_path / "faiss.index"
        
        if docs_file.exists() and index_file.exists():
            with open(docs_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            self.index = faiss.read_index(str(index_file))
            print(f"Loaded {len(self.documents)} documents from index")
    
    def save_index(self):
        """Save index to disk"""
        with open(self.index_path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path / "faiss.index"))
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        doc = docx.Document(docx_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from TXT file"""
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def add_document(self, file_path: str, metadata: Dict = None):
        """Add a document to the store"""
        file_path = Path(file_path)
        
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        # Add chunks to documents
        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk,
                "source": str(file_path),
                "chunk_id": i,
                "metadata": metadata or {}
            }
            self.documents.append(doc)
        
        # Update embeddings and index
        self.update_embeddings()
        print(f"Added {len(chunks)} chunks from {file_path.name}")
    
    def update_embeddings(self):
        """Update embeddings and FAISS index"""
        if not self.documents:
            return
        
        texts = [doc["text"] for doc in self.documents]
        embeddings = embed_model.encode(texts, show_progress_bar=True)
        
        # Create or update FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        self.save_index()
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        if not self.documents or self.index is None:
            return []
        
        query_embedding = embed_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results

# Initialize document store
doc_store = DocumentStore()

# Chat history management
class ChatHistory:
    def __init__(self, history_file="chat_history.json"):
        self.history_file = history_file
        self.sessions = self.load_history()
        self.current_session_id = None
    
    def load_history(self) -> Dict:
        """Load chat history from file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_history(self):
        """Save chat history to file"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.sessions, f, ensure_ascii=False, indent=2)
    
    def new_session(self) -> str:
        """Create a new chat session"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = session_id
        self.sessions[session_id] = {
            "created": datetime.now().isoformat(),
            "messages": []
        }
        return session_id
    
    def add_message(self, role: str, content: str):
        """Add a message to the current session"""
        if self.current_session_id is None:
            self.new_session()
        
        self.sessions[self.current_session_id]["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save_history()
    
    def get_session(self, session_id: str = None) -> List[Dict]:
        """Get messages from a session"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id in self.sessions:
            return self.sessions[session_id]["messages"]
        return []
    
    def list_sessions(self) -> List[Tuple[str, str]]:
        """List all sessions with their creation times"""
        sessions = []
        for sid, data in self.sessions.items():
            created = data.get("created", "Unknown")
            sessions.append((sid, created))
        return sorted(sessions, reverse=True)

# Initialize chat history
chat_history_manager = ChatHistory()

def generate_response(prompt, chat_history, use_rag=True, rag_k=3):
    """
    Generates a response from the PHI-4 model with RAG support.
    """
    # Initialize chat_history if it's None
    if chat_history is None:
        chat_history = []
        chat_history_manager.new_session()
    
    # Save user message to history
    chat_history_manager.add_message("user", prompt)
    
    # Get relevant context from documents if RAG is enabled
    context_docs = []
    if use_rag and doc_store.documents:
        context_docs = doc_store.search(prompt, k=rag_k)
    
    # Build the conversation context
    context = ""
    
    # Add document context if available
    if context_docs:
        context += "### Relevant Information from Knowledge Base:\n"
        for i, doc in enumerate(context_docs, 1):
            context += f"\n[Source {i}: {os.path.basename(doc['source'])}]\n"
            context += doc['text'] + "\n"
        context += "\n### Conversation:\n"
    
    # Add chat history
    for message in chat_history:
        if message["role"] == "user":
            context += "User: " + message["content"] + "\n"
        else:  # role == "assistant"
            context += "Assistant: " + message["content"] + "\n"
    
    # Add current user message
    context += "User: " + prompt + "\n"
    context += "Assistant: "
    
    # Tokenize and generate response
    encoded_inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encoded_inputs.input_ids.to(model.device)
    attention_mask = encoded_inputs.attention_mask.to(model.device)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + 200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(context):].strip()
    
    # Save assistant response to history
    chat_history_manager.add_message("assistant", response)
    
    # Update chat history
    chat_history.append({"role": "user", "content": prompt})
    chat_history.append({"role": "assistant", "content": response})
    
    return chat_history

def upload_document(file):
    """Handle document upload"""
    if file is None:
        return "No file uploaded"
    
    try:
        doc_store.add_document(file.name)
        return f"Successfully added {os.path.basename(file.name)} to knowledge base. Total documents: {len(doc_store.documents)}"
    except Exception as e:
        return f"Error adding document: {str(e)}"

def load_session(session_id):
    """Load a previous chat session"""
    if session_id:
        messages = chat_history_manager.get_session(session_id)
        # Convert to format expected by Gradio
        chat_history = []
        for msg in messages:
            chat_history.append({"role": msg["role"], "content": msg["content"]})
        return chat_history
    return []

def get_session_list():
    """Get list of available sessions"""
    sessions = chat_history_manager.list_sessions()
    return [f"{sid} ({created})" for sid, created in sessions]

# Create Gradio interface
with gr.Blocks(title="PHI-4 RAG Chatbot") as iface:
    gr.Markdown("# PHI-4 Chatbot with RAG and Persistent History")
    gr.Markdown("Chat with Microsoft's PHI-4 model enhanced with document knowledge base and chat history.")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=500)
            msg = gr.Textbox(label="Message", placeholder="Type your message here...")
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Chat")
                new_session = gr.Button("New Session")
            
            with gr.Accordion("RAG Settings", open=False):
                use_rag = gr.Checkbox(label="Use Knowledge Base", value=True)
                rag_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, 
                                 label="Number of relevant documents to retrieve")
        
        with gr.Column(scale=1):
            gr.Markdown("### Document Upload")
            file_upload = gr.File(label="Upload Document", 
                                 file_types=[".pdf", ".txt", ".docx", ".md"])
            upload_btn = gr.Button("Add to Knowledge Base")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            gr.Markdown("### Chat History")
            session_dropdown = gr.Dropdown(label="Load Previous Session", 
                                         choices=get_session_list())
            load_btn = gr.Button("Load Session")
            refresh_btn = gr.Button("Refresh Sessions")
    
    # Event handlers
    def user_submit(message, history, use_rag, rag_k):
        return generate_response(message, history, use_rag, rag_k)
    
    msg.submit(user_submit, [msg, chatbot, use_rag, rag_k], [chatbot])
    submit.click(user_submit, [msg, chatbot, use_rag, rag_k], [chatbot])
    
    clear.click(lambda: [], outputs=[chatbot])
    
    new_session.click(
        lambda: ([], chat_history_manager.new_session(), "New session created"),
        outputs=[chatbot, session_dropdown, upload_status]
    )
    
    upload_btn.click(upload_document, inputs=[file_upload], outputs=[upload_status])
    
    load_btn.click(
        lambda x: load_session(x.split()[0]) if x else [],
        inputs=[session_dropdown],
        outputs=[chatbot]
    )
    
    refresh_btn.click(
        lambda: gr.Dropdown(choices=get_session_list()),
        outputs=[session_dropdown]
    )

if __name__ == "__main__":
    iface.launch()