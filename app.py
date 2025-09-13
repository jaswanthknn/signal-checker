from flask import Flask, render_template, jsonify
import io
import sys
import time
import numpy as np
from ai_model import NetworkPredictor, get_network_latency, find_best_network # make sure scan_networks() exists

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


@app.route("/scan", methods=["GET"])
def scan_networks_route():
    try:
        # capture stdout from your existing ranking print function
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        # run find_best_network()  -> must internally print ranking table
        results = find_best_network()

        sys.stdout = old_stdout
        ranking_output = mystdout.getvalue()

        # send ranking as plain text back to frontend
        return jsonify({"ranking": ranking_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
