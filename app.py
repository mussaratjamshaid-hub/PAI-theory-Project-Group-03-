from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
try:
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model_lr.pkl", "rb"))
except FileNotFoundError:
    print("Warning: Model files not found. Please train the model first.")
    vectorizer = None
    model = None

@app.route("/", methods=["GET"])
def home():
    """Render the main page"""
    return render_template("index.html")

@app.route("/api/classify", methods=["POST"])
def classify():
    """API endpoint for email classification"""
    try:
        data = request.get_json()
        email_text = data.get("text", "").strip()
        
        if not email_text:
            return jsonify({"error": "Email text cannot be empty"}), 400
        
        if model is None or vectorizer is None:
            return jsonify({"error": "Model not trained. Please train the model first."}), 500
        
        # Vectorize and predict
        vec = vectorizer.transform([email_text])
        pred = model.predict(vec)[0]
        confidence = model.predict_proba(vec)[0]
        
        result = {
            "prediction": "Spam" if pred == 1 else "Not Spam",
            "status": "❌" if pred == 1 else "✅",
            "confidence": float(max(confidence)) * 100,
            "spam_probability": float(confidence[1]) * 100
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status", methods=["GET"])
def status():
    """Check API and model status"""
    return jsonify({
        "status": "running",
        "model_loaded": model is not None and vectorizer is not None
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
