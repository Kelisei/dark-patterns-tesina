from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# Fallbacks for specific endpoints using the unified predictor


def _get_predictions(texts):
    from src.predictor.ml_predictor import get_predictor

    predictor = get_predictor()
    return predictor.predict(texts)


@app.post("/scarcity")
def detect_scarcity():
    try:
        json_data = request.get_json()
        texts_info = json_data.get("texts", [])

        raw_texts = [item.get("text", "") for item in texts_info]
        if not raw_texts:
            return {"version": "1.0", "instances": []}

        predictions = _get_predictions(raw_texts)

        instances = []
        for i, item in enumerate(texts_info):
            pred = predictions[i]
            # Verify if 'fake_scarcity' is in the labels list
            has_scarcity = "fake_scarcity" in pred["labels"]

            instances.append(
                {
                    "text": item.get("text", ""),
                    "path": item.get("path", ""),
                    "id": item.get("id", ""),
                    "has_scarcity": has_scarcity,
                }
            )

        return {"version": "1.0", "instances": instances}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.post("/urgency")
def detect_urgency():
    try:
        json_data = request.get_json()
        texts_info = json_data.get("texts", [])

        raw_texts = [item.get("text", "") for item in texts_info]
        if not raw_texts:
            return {"version": "1.0", "urgency_instances": []}

        predictions = _get_predictions(raw_texts)

        instances = []
        for i, item in enumerate(texts_info):
            pred = predictions[i]
            # Verify if 'fake_urgency' is in the labels list
            has_urgency = "fake_urgency" in pred["labels"]

            instances.append(
                {
                    "text": item.get("text", ""),
                    "path": item.get("path", ""),
                    "id": item.get("id", ""),
                    "has_urgency": has_urgency,
                }
            )

        return {"version": "1.0", "urgency_instances": instances}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.post("/shaming")
def detect_shaming():
    try:
        json_data = request.get_json()
        sentences = []

        # Determine the format based on Version
        if json_data.get("Version", "0.1") != "0.2":
            tokens = json_data.get("tokens", [])
            raw_texts = [t.get("text", "") for t in tokens]

            # Use ML model instead of old functions
            if raw_texts:
                predictions = _get_predictions(raw_texts)
                for i, token in enumerate(tokens):
                    if "shaming" in predictions[i]["labels"]:
                        # Old shaming script appended a dict with text and path per detected
                        sentences.append(
                            {
                                "text": token.get("text", ""),
                                "path": token.get("path", ""),
                                "id": token.get("id", ""),
                            }
                        )
            return jsonify(sentences)
        else:
            # Handles V0.2 logic if ever used
            data = json_data.get("texts", [])
            raw_texts = [t.get("text", "") for t in data]

            instances = []
            if raw_texts:
                predictions = _get_predictions(raw_texts)
                for i, t in enumerate(data):
                    instances.append(
                        {
                            "text": t.get("text", ""),
                            "path": t.get("path", ""),
                            "id": t.get("id", ""),
                            "has_shaming": "shaming" in predictions[i]["labels"],
                        }
                    )
            return {"version": "0.2", "instances": instances}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.post("/detect")
def detect_dark_patterns():
    try:
        json_data = request.get_json()
        texts_info = json_data.get("texts", [])

        if not texts_info:
            texts_info = json_data.get("tokens", [])
            if not texts_info:
                return {"version": "1.0", "instances": []}

        raw_texts = [item.get("text", "") for item in texts_info]
        predictions = _get_predictions(raw_texts)

        instances = []
        for i, item in enumerate(texts_info):
            pred = predictions[i]
            instances.append(
                {
                    "text": item.get("text", ""),
                    "path": item.get("path", ""),
                    "id": item.get("id", ""),
                    "detected": pred["detected"],
                    "labels": pred["labels"],
                }
            )

        return {"version": "1.0", "instances": instances}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
