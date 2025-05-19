from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import cv2
from pyzbar.pyzbar import decode
import requests
import numpy as np

app = Flask(__name__, static_folder="public", static_url_path="")
CORS(app, resources={r"/detect": {"origins": "http://localhost:3000"}})

# Load phishing URL detection model
phishing_model_name = "imanoop7/bert-phishing-detector"
phishing_tokenizer = AutoTokenizer.from_pretrained(phishing_model_name)
phishing_model = AutoModelForSequenceClassification.from_pretrained(phishing_model_name)

# Load phishing email detection model
email_model_name = "ElSlay/BERT-Phishing-Email-Model"
email_tokenizer = AutoTokenizer.from_pretrained(email_model_name)
email_model = AutoModelForSequenceClassification.from_pretrained(email_model_name)
email_classifier = pipeline(
    "text-classification", model=email_model, tokenizer=email_tokenizer
)


def predict_phishing_url(url: str) -> str:
    inputs = phishing_tokenizer(url, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = phishing_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    phishing_prob = probs[0][1].item()
    if phishing_prob > 0.5:
        return "Phishing: This URL seems suspicious."
    else:
        return "Legitimate: This URL appears safe."


def predict_fake_or_real_email_content(text: str) -> str:
    result = email_classifier(text)[0]
    label = result["label"]
    score = result["score"]
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


@app.route("/")
def serve_homepage():
    # Serve the HomePage.html from the public folder
    return send_from_directory(app.static_folder, "HomePage.html")


@app.route("/MainHub")
def serve_mainhub():
    # Serve the MainHub.html from the public folder
    return send_from_directory(app.static_folder, "MainHub.html")


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


# NEW ROUTE: Accept POST requests at /MainHub/submit
@app.route("/MainHub/submit", methods=["POST"])
def mainhub_submit():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        input_type = data.get("type")
        user_input = data.get("input")

        if not input_type or not user_input:
            return jsonify({"error": "Missing 'type' or 'input' fields"}), 400

        valid_types = {"URL", "Email", "Qr code"}
        if input_type not in valid_types:
            return (
                jsonify(
                    {
                        "error": f"Invalid type '{input_type}'. Expected one of {valid_types}"
                    }
                ),
                400,
            )

        # Run phishing detection based on input type
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
    # Run on port 5000, accessible on all interfaces
    app.run(host="0.0.0.0", port=5000, debug=False)
