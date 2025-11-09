from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("Svc.pkl", "rb") as f:
    model = pickle.load(f)

CODE2LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        math = float(request.form["math"])
        reading = float(request.form["reading"])
        writing = float(request.form["writing"])

        X = np.array([[math, reading, writing]], dtype=float)
        pred = model.predict(X)[0]

        # Sayısal tahminleri harfe dönüştür
        if isinstance(pred, (int, np.integer)) and pred in CODE2LETTER:
            pred_label = CODE2LETTER[int(pred)]
        else:
            pred_label = str(pred).upper()

        result = f"Predicted Group: {pred_label}"
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
