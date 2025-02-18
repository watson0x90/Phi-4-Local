import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Specify the model name
model_name = "microsoft/phi-4"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def generate_response(prompt, chat_history):
    """
    Generates a response from the PHI-4 model using an updated chat history format.
    Chat history is now a list of dictionaries with "role" and "content" keys.
    """
    # Initialize chat_history if it's None.
    if chat_history is None:
        chat_history = []

    # Append the user's message using the new format.
    chat_history.append({"role": "user", "content": prompt})
    
    # Build the conversation context by iterating over the chat history.
    context = ""
    for message in chat_history:
        if message["role"] == "user":
            context += "User: " + message["content"] + "\n"
        else:  # role == "assistant"
            context += "Assistant: " + message["content"] + "\n"
    # Append the assistant prompt.
    context += "Assistant: "
    
    # Tokenize the context and prepare the attention mask.
    encoded_inputs = tokenizer(context, return_tensors="pt")
    input_ids = encoded_inputs.input_ids.to(model.device)
    attention_mask = encoded_inputs.attention_mask.to(model.device)
    
    # Generate a response from the model.
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + 100,  # Adjust response length as needed.
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated tokens.
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response.
    response = generated_text[len(context):].strip()
    
    # Append the assistant's response to the chat history.
    chat_history.append({"role": "assistant", "content": response})
    
    # Return the updated chat history.
    return chat_history

# Create a Gradio ChatInterface using the "messages" type.
iface = gr.ChatInterface(
    fn=generate_response,
    type="messages",  # Set to "messages" to use the new chat message format.
    title="PHI-4 Chatbot",
    description="Chat with Microsoft's PHI-4 model running locally."
)

if __name__ == "__main__":
    iface.launch()
