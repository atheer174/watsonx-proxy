from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
generator = pipeline("text-generation", model="gpt2")

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt", "")
    result = generator(prompt, max_length=100)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
