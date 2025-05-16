from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import cv2
from pyzbar.pyzbar import decode
import requests
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/detect": {"origins": "http://localhost:3000"}})

# URL & QR Code Phishing Detection Model
phishing_model_name = "imanoop7/bert-phishing-detector"
phishing_tokenizer = AutoTokenizer.from_pretrained(phishing_model_name)
phishing_model = AutoModelForSequenceClassification.from_pretrained(phishing_model_name)

# Email Phishing Detection Model
email_model_name = "ElSlay/BERT-Phishing-Email-Model"
email_tokenizer = AutoTokenizer.from_pretrained(email_model_name)
email_model = AutoModelForSequenceClassification.from_pretrained(email_model_name)
email_classifier = pipeline(
    "text-classification", model=email_model, tokenizer=email_tokenizer
)

# ================================
# Helper Functions
# ================================


def predict_phishing_url(url: str) -> str:
    inputs = phishing_tokenizer(url, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = phishing_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    phishing_prob = probs[0][1].item()
    if phishing_prob > 0.5:
        return f"Phishing: This URL seems suspicious."
    return f"Legitimate: This URL appears safe."


def predict_fake_or_real_email_content(text: str) -> str:
    result = email_classifier(text)[0]
    label = result["label"]
    score = result["score"]

    # Adjust this condition based on actual labels you get from your model
    if label in ["NEGATIVE", "LABEL_1"] and score >= 0.5:
        return "Scam/Fake: Message seems suspicious based on tone and content."
    return "Real/Legitimate: Message appears safe."


def phishing_qr_url(url: str) -> float:
    inputs = phishing_tokenizer(url, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = phishing_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs[0][1].item()


def extract_url_from_qr(image_path_or_url: str) -> str | None:
    try:
        if image_path_or_url.startswith(("http://", "https://")):
            response = requests.get(image_path_or_url)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path_or_url)
            if image is None:
                return None
        decoded_objects = decode(image)
        for obj in decoded_objects:
            return obj.data.decode("utf-8")
    except Exception:
        return None
    return None


@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json
        input_type = data.get("type")
        user_input = data.get("input")

        if input_type == "URL":
            output = predict_phishing_url(user_input)
        elif input_type == "Email":
            output = predict_fake_or_real_email_content(user_input)
        elif input_type == "Qr code":
            extracted_url = extract_url_from_qr(user_input)
            if extracted_url:
                phishing_prob = phishing_qr_url(extracted_url)
                output = f"Phishing probability: {phishing_prob:.2f}"
            else:
                output = "Error: Could not extract URL from QR code."
        else:
            output = "Error: Invalid input type."

        return jsonify({"result": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
