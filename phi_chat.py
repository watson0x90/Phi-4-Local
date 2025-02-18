from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

# Specify the model name
model_name = "microsoft/phi-4"

# Load the tokenizer and model from Hugging Face
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def generate_response(prompt, chat_history=""):
    """
    Combines the chat history with the new prompt, generates a response,
    and returns the updated conversation.
    """
    # For a simple implementation, we’ll just consider the current prompt.
    # For more context-aware conversations, consider incorporating `chat_history`.
    input_text = chat_history + "\nUser: " + prompt + "\nAssistant: "
    
    # Tokenize input and move to the appropriate device
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 100,  # adjust max_length as needed
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated text and extract the new assistant response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the new response (this assumes that the generated text appends the assistant's reply)
    response = generated_text[len(input_text):].strip()
    
    # Update chat history
    updated_history = input_text + response
    return updated_history, updated_history

# Create a Gradio interface using a chatbot layout
iface = gr.ChatInterface(
    fn=generate_response,
    title="PHI-4 Chatbot",
    description="Chat with Microsoft's PHI-4 model running locally."
)

if __name__ == "__main__":
    iface.launch()
