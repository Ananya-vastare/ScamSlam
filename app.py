from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib  # added for loading traditional ML models

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)
# Load the url phishing model

url_model_name = "Adnan-AI-Labs/URLShield-DistilBERT"  # Example placeholder, replace with actual model
tokenizer = AutoTokenizer.from_pretrained(url_model_name)
model = AutoModelForSequenceClassification.from_pretrained(url_model_name)
url_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


# Load phishing email detection model (transformer)
email_model_name = "ElSlay/BERT-Phishing-Email-Model"
email_tokenizer = AutoTokenizer.from_pretrained(email_model_name)
email_model = AutoModelForSequenceClassification.from_pretrained(email_model_name)
email_classifier = pipeline(
    "text-classification", model=email_model, tokenizer=email_tokenizer
)

# Load traditional ML model and vectorizer for SMS phishing detection
sms_vectorizer = joblib.load("sms_vectorizer.pkl")  # path to your saved vectorizer
sms_model = joblib.load("sms_model.pkl")  # path to your saved ML model


def predict_phishing_url(url: str) -> str:
    result = url_classifier(url)[0]
    label = result["label"].lower()
    score = result["score"]
    if "phish" in label and score > 0.6:
        return "This URL is Phished"
    else:
        return "This URL is not Phished"


def predict_fake_or_real_email_content(text: str) -> str:
    result = email_classifier(text)[0]
    label = result["label"]
    score = result["score"]
    if label in ["NEGATIVE", "LABEL_1"] and score >= 0.5:
        return "This email is Phished"
    return "This email is not Phished"


def predict_sms_phishing(text: str) -> str:
    # Vectorize the input text
    X = sms_vectorizer.transform([text])
    # Predict the label (assuming 1 = phishing/spam, 0 = not phishing)
    pred = sms_model.predict(X)[0]
    # Predict probability if supported
    prob = (
        sms_model.predict_proba(X)[0][1]
        if hasattr(sms_model, "predict_proba")
        else None
    )

    if pred == 1:
        if prob is not None:
            return "This SMS is Phished"
        else:
            return "This SMS is Phished"
    else:
        if prob is not None:
            return f"This SMS is not Phished  "
        else:
            return "This SMS is not Phished"


@app.route("/")
@app.route("/Homepage.html")
def homepage():
    return render_template("Homepage.html")


@app.route("/MainHub.html")
def mainhub():
    return render_template("MainHub.html")


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
        elif input_type == "SMS":
            output = predict_sms_phishing(user_input)
        else:
            output = "Error: Invalid input type."

        return jsonify({"result": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/MainHub/submit", methods=["POST"])
def mainhub_submit():
    try:
        if request.content_type and request.content_type.startswith(
            "multipart/form-data"
        ):
            # No file upload handling needed anymore, since no QR code
            return (
                jsonify({"error": "File upload not supported in SMS detection."}),
                400,
            )
        else:
            data = request.json
            input_type = data.get("type")
            user_input = data.get("input")

            if not input_type or not user_input:
                return jsonify({"error": "Missing type or input"}), 400

            if input_type == "URL":
                output = predict_phishing_url(user_input)
            elif input_type == "Email":
                output = predict_fake_or_real_email_content(user_input)
            elif input_type == "SMS":
                output = predict_sms_phishing(user_input)
            else:
                output = "Error: Invalid input type."

            return jsonify({"result": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
