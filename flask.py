from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from threading import Thread
from pyngrok import ngrok

app = Flask(__name__)

# Load the model and tokenizer
model_path = "./fine_tuned_bartDataset_model_all_outputs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path).to(device)

# Load the HTML template
with open("index.html", "r") as file:
    html_template = file.read()

# Define the generate function
def generate_output(input_text, max_length=1024):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    ).to(device)

    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_output

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/generate', methods=['POST'])
def generate():
    input_data = request.json
    input_text = input_data.get("input", "")

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    print(f"Received input text: {input_text}")  # Debugging
    generated_output = generate_output(input_text)
    print(f"Generated output: {generated_output}")  # Debugging
    return jsonify({"generated_output": generated_output})

# To run the Flask app in the background
def run_app():
    app.run(port=5000)

thread = Thread(target=run_app)
thread.start()

# Expose the Flask app with ngrok
ngrok.kill()
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")