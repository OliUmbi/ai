from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import uuid
import time

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "http://localhost", "methods": "GET, POST, PUT, DELETE"}})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_name = "Qwen/Qwen2-0.5B-Instruct"
text_model = None
text_tokenizer = None

image_name = "stabilityai/stable-diffusion-2-1"
image_model = None


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "text": {
            "name": text_name,
            "active": text_active()
        },
        "image": {
            "name": image_name,
            "active": image_active()
        }
    })


@app.route("/configure", methods=["POST"])
def configure():
    body = request.get_json()

    if body["text"]["active"]:
        image_deactivate()
        text_activate()
    else:
        text_deactivate()

    if body["image"]["active"]:
        text_deactivate()
        image_activate()
    else:
        image_deactivate()

    return status()


def text_active():
    global text_model, text_tokenizer

    return text_model is not None and text_tokenizer is not None


def text_activate():
    global text_model, text_tokenizer

    text_model = AutoModelForCausalLM.from_pretrained(text_name).to(device)
    text_tokenizer = AutoTokenizer.from_pretrained(text_name)


def text_deactivate():
    global text_model, text_tokenizer

    text_model = None
    text_tokenizer = None


@app.route("/text/generate", methods=["POST"])
def text_generate():
    global text_model, text_tokenizer

    if not text_active():
        return jsonify({
            "message": "Text model is not active"
        }), 400

    body = request.get_json()

    duration = time.time()

    messages = [
        {"role": "system", "content": body["instruction"]},
        {"role": "user", "content": body["prompt"]}
    ]
    chat = text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = text_tokenizer(chat, return_tensors="pt", padding=True).to(device)

    generated_ids = text_model.generate(
        model_inputs.input_ids,
        max_new_tokens=body["tokens"],
        num_return_sequences=1,
        do_sample=True,
        temperature=body["temperature"]
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = text_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return jsonify({
        "instruction": body["instruction"],
        "prompt": body["prompt"],
        "response": response,
        "duration": time.time() - duration,
        "tokens": body["tokens"],
        "temperature": body["temperature"]
    })


def image_active():
    global image_model

    return image_model is not None


def image_activate():
    global image_model

    image_model = StableDiffusionPipeline.from_pretrained(image_name, torch_dtype=torch.float16).to(device)
    image_model.scheduler = DPMSolverMultistepScheduler.from_config(image_model.scheduler.config)


def image_deactivate():
    global image_model

    image_model = None


@app.route("/image/generate", methods=["POST"])
def image_generate():
    global image_model

    if not image_active():
        return jsonify({
            "message": "Image model is not active"
        }), 400

    body = request.get_json()

    duration = time.time()
    image = str(uuid.uuid4())

    with torch.no_grad():
        image_model(body["prompt"], width=body["width"], height=body["height"]).images[0].save("../images/" + image + ".png")

    return jsonify({
        "prompt": body["prompt"],
        "image": image,
        "duration": time.time() - duration,
        "width": body["width"],
        "height": body["height"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
