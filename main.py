import gradio as gr
import ollama
import sys

# --- Configuration ---
# Set the model you want to use.
# Make sure you have this model pulled in Ollama (e.g., `ollama pull llama3`)
DEFAULT_MODEL = 'qwen3:1.7b'  # <<< --- CHANGE THIS to your preferred model


# --- Ollama Chat Function ---
# This is the core function that Gradio's ChatInterface will call.
# 'message' is the new user input (str).
# 'history' is a list of past (user, assistant) message pairs (list[list[str, str]]).
def chat_with_ollama(message, history):
    """
    Handles the chat logic, communicating with the Ollama API.
    Streams the response back to the Gradio interface.
    """
    try:
        # 1. Convert Gradio's history format to Ollama's messages format
        # Gradio history: [ [user_msg_1, bot_msg_1], [user_msg_2, bot_msg_2], ... ]
        messages = []
        for user_msg, bot_msg in history:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': bot_msg})

        # 2. Add the new user message
        messages.append({'role': 'user', 'content': message})

        # 3. Call the Ollama API with streaming
        # `model` is set to the one you configured
        # `stream=True` is crucial for the streaming effect
        stream = ollama.chat(
            model=DEFAULT_MODEL,
            messages=messages,
            stream=True
        )

        # 4. Stream the response back to Gradio
        response = ""
        for chunk in stream:
            # `chunk['message']['content']` contains the next piece of text
            part = chunk['message']['content']
            response += part
            # `yield response` sends the *updated* full response to Gradio
            # This creates the "typing" effect
            yield response

    except Exception as e:
        # Handle potential errors (e.g., Ollama not running, model not found)
        print(f"Error occurred: {e}", file=sys.stderr)
        # Yield a user-friendly error message
        error_message = f"An error occurred: {e}. \n\n- Is the Ollama service running? \n- Is the model '{DEFAULT_MODEL}' pulled and spelled correctly?"
        yield error_message


# --- Gradio UI Definition ---
# This sets up the web interface.
print("Building Gradio UI...")

# We use gr.ChatInterface, which provides a beautiful, ready-made chat UI.
# We simply pass our `chat_with_ollama` function to it.
demo = gr.ChatInterface(
    fn=chat_with_ollama,  # The function to call on user input
    title="Chat with Ollama",
    description=f"A simple, beautiful chat UI for your local Ollama models. Currently using: **{DEFAULT_MODEL}**.",

    # UI element customization
    theme="soft",  # Try "default", "soft", "glass", or "dark"
    chatbot=gr.Chatbot(
        height=600,
        avatar_images=("https://placehold.co/40x40/E8F5E9/4CAF50?text=U",
                       "https://placehold.co/40x40/F3E5F5/8E24AA?text=AI")
    ),
    textbox=gr.Textbox(
        placeholder="Ask me anything...",
        container=False,
        scale=7  # Make the textbox wider
    ),

    # Button labels - REMOVED as they cause errors in newer Gradio versions
    # clear_btn="Clear Chat",
    # retry_btn="Retry Last",
    # undo_btn="Undo",

    # Example questions to show the user
    examples=[
        "What is the capital of France?",
        "Write a short poem about a robot",
        "Explain the concept of recursion in programming"
    ]
)

# --- Run the App ---
if __name__ == "__main__":
    print("Starting Gradio UI for Ollama...")
    print(f"Your Ollama model is set to: {DEFAULT_MODEL}")
    print("Access the chat UI in your browser at the URL shown below (e.g., http://127.0.0.1:7860)")

    # `demo.launch()` starts the web server
    # `server_name="0.0.0.0"` makes it accessible on your local network (optional)
    # `share=True` would create a public link (use with caution)
    demo.launch(server_name="0.0.0.0")