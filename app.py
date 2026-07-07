from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/placement_model.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = float(request.form["gender"])
    ssc_p = float(request.form["ssc_p"])
    ssc_b = float(request.form["ssc_b"])
    hsc_p = float(request.form["hsc_p"])
    hsc_b = float(request.form["hsc_b"])
    hsc_s = float(request.form["hsc_s"])
    degree_p = float(request.form["degree_p"])
    degree_t = float(request.form["degree_t"])
    workex = float(request.form["workex"])
    etest_p = float(request.form["etest_p"])
    specialisation = float(request.form["specialisation"])
    mba_p = float(request.form["mba_p"])

    features = np.array([[
        gender,
        ssc_p,
        ssc_b,
        hsc_p,
        hsc_b,
        hsc_s,
        degree_p,
        degree_t,
        workex,
        etest_p,
        specialisation,
        mba_p
    ]])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    if prediction[0] == 1:
        result = "Placed"
    else:
        result = "Not Placed"

    return render_template(
        "index.html",
        prediction_text=f"Prediction: {result}"
    )


if __name__ == "__main__":
    app.run(debug=True)