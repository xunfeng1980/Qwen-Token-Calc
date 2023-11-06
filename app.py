import uuid

import gradio as gr

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen",
    trust_remote_code=True,
    resume_download=True,
)

with (gr.Blocks() as demo):
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])


    def respond(message, chat_history):
        t = tokenizer(message)
        input_ids = t['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        out = []
        for o in tokens:
            out.append(o.decode("utf-8", errors='replace'))

        chat_history.append((message, f"tokens: {str(len(t['input_ids']))}"))
        chat_history.append((None, str(out)))
        return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(server_name='0.0.0.0', share=False)
