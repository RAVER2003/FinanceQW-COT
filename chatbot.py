import os
import json
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
import re


# -------------------------
# Model Load
# -------------------------
MODEL_PATH = "qwen-final-model"
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, )
model.eval()


# -------------------------
# Chat Storage Utilities
# -------------------------
CHAT_DIR = "chat_sessions"
os.makedirs(CHAT_DIR, exist_ok=True)
context_length = 4   # Sliding window for model prompt


def sanitize_name(text):
    """Generate a safe filename from the first prompt."""
    text = text.strip().lower()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return text[:40]  # keep file short


def get_chat_path(chatname):
    return os.path.join(CHAT_DIR, f"{chatname}.json")


def load_chat_list():
    files = os.listdir(CHAT_DIR)
    chats = [f.replace(".json", "") for f in files if f.endswith(".json")]
    return chats


def load_chat(chatname):
    """Load chat JSON file."""
    path = get_chat_path(chatname)
    if not os.path.exists(path):
        return []
    return json.load(open(path, "r"))


def save_chat(chatname, messages):
    """Save entire chat history to JSON."""
    path = get_chat_path(chatname)
    json.dump(messages, open(path, "w"), indent=2)


# -------------------------
# Prompt Formatting
# -------------------------
def build_prompt(query, history_list):
    system_msg = "You are a helpful financial reasoning assistant."

    prompt = f"Instruction: {system_msg}\n"
    recent = history_list[-context_length:]  # sliding window

    for msg in recent:
        prompt += f"Input: {msg['user']}\n"
        prompt += f"Output: {msg['assistant']}\n"

    prompt += f"Input: {query}\nOutput:"
    return prompt



def split_think_answer(txt):
    think, ans = txt , ""
    if "<think>" in txt and "</think>" in txt:
        think = txt.split("<think>")[1].split("</think>")[0]
        ans = txt.split("</think>")[-1].strip()
    return think, ans

# -------------------------
# Streaming Chat Function
# -------------------------
def chat_fn(user_msg, selected_chat, chatbot_ui):

    # --------------------------------------------
    # Step 1: Determine chat ID
    # --------------------------------------------
    if selected_chat == "New Chat":
        # Create new chat using first user query as name
        new_name = sanitize_name(user_msg)
        if new_name == "":
            new_name = f"chat_{len(load_chat_list())+1}"
        selected_chat = new_name

        # Initialize empty chat file
        save_chat(selected_chat, [])

        # Update dropdown immediately
        yield gr.update(choices=["New Chat"] + load_chat_list(), value=selected_chat), chatbot_ui, "", ""

    # Load chat history
    history = load_chat(selected_chat)

    # --------------------------------------------
    # Step 2: Build prompt with sliding context
    # --------------------------------------------
    prompt = build_prompt(user_msg, history)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_new_tokens=800,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
            streamer=streamer
        )
    )
    thread.start()

    full_output = ""
    thinking = ""
    answer = ""

    # --------------------------------------------
    # Step 3: Stream tokens
    # --------------------------------------------
    for token in streamer:
        full_output += token

        thinking, answer = split_think_answer(full_output)

        # Update streamed UI
        new_chat_history = chatbot_ui + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer}
        ]

        yield (
            gr.update(choices=["New Chat"] + load_chat_list(), value=selected_chat),
            new_chat_history,
            thinking,
            answer
        )


    history.append({"user": user_msg, "assistant": answer})
    save_chat(selected_chat, history)



def load_chat_clicked(chatname):
    if chatname == "New Chat":
        return [], None,None
    history = load_chat(chatname)

    # Convert to Chatbot UI format
    chat_ui = []
    for msg in history:
        chat_ui.append({"role": "user", "content": msg["user"]})
        chat_ui.append({"role": "assistant", "content": msg["assistant"]})

    return chat_ui, "",""


# -------------------------
# GUI Layout
# -------------------------
with gr.Blocks() as demo:

    gr.Markdown("<h2> FinanceQW</h2>")

    with gr.Row():
        with gr.Column(scale=1):
            chat_selector = gr.Dropdown(
                label="Saved Chats",
                choices=["New Chat"] + load_chat_list(),
                value="New Chat"
            )
            load_btn = gr.Button("Load Chat")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Conversation")
            user_box = gr.Textbox(label="Your Message")
            send_btn = gr.Button("Send")

        with gr.Column(scale=2):
            thinking_box = gr.Textbox(label="Model Thinking", lines=10)
            answer_box = gr.Textbox(label="Final Answer", lines=10)

    # Load chat action
    load_btn.click(
        load_chat_clicked,
        inputs=[chat_selector],
        outputs=[chatbot, thinking_box,answer_box]
    )

    # Chat action
    send_btn.click(
        chat_fn,
        inputs=[user_box, chat_selector, chatbot],
        outputs=[chat_selector, chatbot, thinking_box, answer_box],
        queue=True
    )


demo.queue().launch()
