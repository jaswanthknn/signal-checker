from flask import Flask, render_template, request, jsonify
from ai_model import NetworkPredictor, get_network_latency
import time
import numpy as np

app = Flask(__name__)

# Load model once on server start
try:
    predictor = NetworkPredictor()
except FileNotFoundError:
    predictor = None
    print("⚠ Model files not found. Please train using '--step all' before running Flask app.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_network_status():
    if not predictor:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    if not data:  # protect against None
        return jsonify({"error": "Invalid JSON body."}), 400

    duration = int(data.get("duration", 10))          # default 10
    sampling_rate = int(data.get("sampling_rate", 2)) # default 2
    n_samples = duration * sampling_rate

    latency_signal = []
    for _ in range(n_samples):
        latency = get_network_latency()
        latency_signal.append(latency if latency is not None else 1000.0)
        time.sleep(1 / sampling_rate)

    if len(latency_signal) < 10:
        return jsonify({"error": "Not enough data."}), 400

    try:
        result_text = predictor.predict(np.array(latency_signal))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prediction": result_text,
        "latency_signal": latency_signal
    })

if __name__ == "__main__":
    app.run(debug=True)
