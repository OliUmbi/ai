import flask
from flask import Flask, request, jsonify, make_response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


@app.route("/generate", methods=["OPTIONS"])
def preflight():
    response = make_response()
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "OPTIONS, GET, POST, PUT, DELETE")
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:81")
    return response


@app.route("/generate", methods=["POST"])
def generate():
    instruction = request.json.get("instruction", "You are a helpful assistant.")
    prompt = request.json.get("prompt", "Hello")
    tokens = request.json.get("tokens", 100)
    temperature = request.json.get("temperature", 0.7)

    start = time.time()

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=tokens,
        num_return_sequences=1,
        do_sample=True,
        temperature=temperature
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    duration = time.time() - start

    response = jsonify({
        "instruction": instruction,
        "prompt": prompt,
        "response": response,
        "duration": duration,
        "tokens": tokens,
        "temperature": temperature
    })
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "OPTIONS, GET, POST, PUT, DELETE")
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:81")
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
