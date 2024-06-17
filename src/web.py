from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

app = Flask(__name__)

device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
    torch_dtype=torch.float32,
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")


def generate_response(prompt):
    start_time = time.time()

    messages = [
        {"role": "system", "content": "You are a helpful assistant and only give very short one sentence answers."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=25,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        early_stopping=True,
        temperature=0.7

    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"time: {time.time() - start_time:.2f} seconds")
    return response


@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get('prompt', '')
    response = generate_response(prompt)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
