import pickle

import pandas as pd
from flask import Flask, render_template, request

from features import FEATURES

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Field label, unit and an example value (a sample row from the dataset)
FIELDS = [
    ("ph", "pH", "0-14", 7.08),
    ("Hardness", "Hardness", "mg/L", 196.4),
    ("Solids", "Total dissolved solids", "ppm", 22014.1),
    ("Chloramines", "Chloramines", "ppm", 7.12),
    ("Sulfate", "Sulfate", "mg/L", 333.8),
    ("Conductivity", "Conductivity", "μS/cm", 426.2),
    ("Organic_carbon", "Organic carbon", "ppm", 14.28),
    ("Trihalomethanes", "Trihalomethanes", "μg/L", 66.4),
    ("Turbidity", "Turbidity", "NTU", 3.97),
]


@app.route("/", methods=["GET", "POST"])
def home():
    result, error, values = None, None, {}

    if request.method == "POST":
        try:
            values = {name: float(request.form[name]) for name in FEATURES}
        except (KeyError, ValueError):
            error = "Please enter a number in every field."
        else:
            row = pd.DataFrame([values], columns=FEATURES)
            proba = model.predict_proba(row)[0][1]
            result = {"potable": proba >= 0.5, "confidence": round(max(proba, 1 - proba) * 100, 1)}

    return render_template("index.html", fields=FIELDS, values=values, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
