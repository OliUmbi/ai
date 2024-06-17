import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cpu")

model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def generate_text(prompt, model, tokenizer, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )

    return [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]


prompt = "im bored, give me ideas what to do"
generated_texts = generate_text(prompt, model, tokenizer, max_length=100, num_return_sequences=1)

for i, text in enumerate(generated_texts):
    print(f"Generated Text {i + 1}:\n{text}\n")
