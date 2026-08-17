import os, gradio as gr, tiktoken
from openai import OpenAI

client = OpenAI()                      # reads OPENAI_API_KEY from environment
MODEL = "gpt-4o-mini"
PIN, POUT = 0.15, 0.60                 # $ per 1M input / output tokens

try:
    ENC = tiktoken.encoding_for_model(MODEL)
except Exception:
    ENC = tiktoken.get_encoding("o200k_base")

def generate(system, prompt, temperature, top_p, max_tokens, fpen, ppen):
    if not prompt.strip():
        return "Please enter a prompt.", ""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature, top_p=top_p, max_tokens=int(max_tokens),
        frequency_penalty=fpen, presence_penalty=ppen)
    u = r.usage
    cost = u.prompt_tokens/1e6*PIN + u.completion_tokens/1e6*POUT
    return r.choices[0].message.content, f"total tokens: {u.total_tokens} | cost: ${cost:.6f}"

def explore(text):
    ids = ENC.encode(text or "")
    pieces = [ENC.decode([i]) for i in ids]
    return (f"Token count: {len(ids)}\nCost as input: ${len(ids)/1e6*PIN:.6f}\n\n"
            f"Tokens: " + "|".join(pieces))

with gr.Blocks(title="LLM Playground & Token Explorer") as demo:
    gr.Markdown("# LLM Playground & Token Explorer")
    with gr.Tab("LLM Playground"):
        s = gr.Textbox(label="System prompt", value="You are a helpful Amazon assistant.")
        p = gr.Textbox(label="Prompt")
        with gr.Row():
            t = gr.Slider(0, 2, value=0.7, step=0.1, label="temperature")
            tp = gr.Slider(0.1, 1, value=1.0, step=0.05, label="top_p")
            mt = gr.Slider(50, 800, value=300, step=50, label="max_tokens")
        with gr.Row():
            fp = gr.Slider(-2, 2, value=0.0, step=0.1, label="frequency_penalty")
            pp = gr.Slider(-2, 2, value=0.0, step=0.1, label="presence_penalty")
        gr.Button("Generate").click(generate, [s, p, t, tp, mt, fp, pp],
                                    [gr.Textbox(label="Response", lines=8),
                                     gr.Textbox(label="Tokens & cost")])
    with gr.Tab("Token Explorer"):
        tx = gr.Textbox(label="Text", value="Amazon Prime delivers unbelievably fast!")
        gr.Button("Explore tokens").click(explore, tx, gr.Textbox(label="Tokens", lines=10))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))