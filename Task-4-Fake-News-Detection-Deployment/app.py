
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained ML pipeline
model = joblib.load("fake_news_model.joblib")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form.get("news", "").strip()

        if news_text:
            result = model.predict([news_text])[0]
            prediction = "Fake News" if result == 0 else "Real News"

    return render_template(
        "index.html",
        prediction=prediction,
        news_text=news_text
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    news_text = data.get("news", "").strip()

    if not news_text:
        return jsonify({"error": "Please provide news text."}), 400

    result = int(model.predict([news_text])[0])
    prediction = "Fake News" if result == 0 else "Real News"

    return jsonify({
        "prediction": prediction,
        "label": result
    })


if __name__ == "__main__":
    app.run(debug=True)
