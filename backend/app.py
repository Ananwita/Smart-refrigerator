from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

# Import your pipeline
from smart_fridge_webcam import load_models, run_pipeline

app = Flask(__name__)
CORS(app)

# 🔥 Load models ONCE at startup
yolo_model, fresh_model = load_models(
    "models/yolo_fruit3.pt",
    "models/freshness_model.pt"
)

print("✅ Models loaded successfully")

# -------------------------------
# Route: Home (optional test)
# -------------------------------
@app.route("/")
def home():
    return "Smart Grocery Backend Running 🚀"

# -------------------------------
# Route: Predict
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image file
        file = request.files["image"]

        # Convert to OpenCV image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run full pipeline
        annotated, results = run_pipeline(frame, yolo_model, fresh_model)

        # Convert results into clean JSON
        output = []
        for r in results:
            output.append({
                "item": r["class_name"],
                "freshness": r["freshness_label"],
                "confidence": r["det_confidence"],
                "freshness_score": r["freshness_score"],
                "days_remaining": r["days_remaining"]
            })

        return jsonify({
            "success": True,
            "results": output
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)