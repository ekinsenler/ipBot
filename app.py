import argparse
import json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


def load_model_and_tokenizer(model_dir):
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos token
    return model, tokenizer


@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        input_text = data['input_text']

        # Encode input text
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)

        # Generate response
        outputs = model.generate(inputs, attention_mask=attention_mask, max_length=50, num_return_sequences=1,
                                 pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"generated_text": generated_text})
    except Exception as e:
        print(f"Error during text generation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deploy the fine-tuned LLaMA model.")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of the fine-tuned model.")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    app.run(host='0.0.0.0', port=5000, debug=True)
