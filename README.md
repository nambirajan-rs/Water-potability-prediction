# Water Potability Prediction 💧

Predicts whether a water sample is safe to drink from 9 water quality measurements. It includes the full exploratory analysis notebook (7 classifiers compared) and a Flask web app serving the final Random Forest model.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Flask](https://img.shields.io/badge/Flask-web%20app-lightgrey)

## Dataset

[Water Quality dataset (Kaggle)](https://www.kaggle.com/datasets/adityakadiwal/water-potability): 3,276 samples, 9 numeric features, binary target `Potability` (61% not potable / 39% potable).

| Feature | Unit | Feature | Unit |
|---|---|---|---|
| pH | 0–14 | Conductivity | μS/cm |
| Hardness | mg/L | Organic carbon | ppm |
| Solids (TDS) | ppm | Trihalomethanes | μg/L |
| Chloramines | ppm | Turbidity | NTU |
| Sulfate | mg/L | | |

`ph`, `Sulfate` and `Trihalomethanes` have missing values (15%, 24% and 5%).

## Project structure

```
notebook/
  Water-potability-prediction-PML.ipynb   EDA, visualisation, 7 models compared
  water_potability_analysis.py            script export of the notebook
data/water_potability.csv
train.py        trains the production model (imputer + scaler + Random Forest pipeline)
features.py     feature order shared by training and the app
app.py          Flask app
templates/index.html
model.pkl       trained pipeline
```

## Run the app

```bash
pip install -r requirements.txt
python app.py        # open http://127.0.0.1:5000, click "Fill example", then "Predict"
```

Retrain: `python train.py`

## Results

**Notebook (model comparison):** Logistic Regression, KNN, Decision Tree, Gaussian/Bernoulli Naive Bayes, Bagging and Random Forest were compared. Random Forest scored best at **79.3%** accuracy.

**Production model (`train.py`): 67.7% test accuracy.** That's lower than the notebook on purpose:

- In the notebook, missing values were filled with the mean **per Potability class**. That uses the label to build the features, which leaks the answer into the training data and inflates accuracy. It also can't be done at prediction time, when the label is unknown.
- `train.py` splits the data first, then fills missing values with the **training-set median** inside a scikit-learn `Pipeline`. The test score is therefore what you'd get on new samples.
- The saved pipeline also includes the `StandardScaler`, so the web app scales inputs exactly as in training. (The first version of the app passed raw values to a model trained on scaled data.)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not potable | 0.70 | 0.83 | 0.76 |
| Potable | 0.62 | 0.43 | 0.51 |

68% is in line with what's typically reported for this dataset: the features overlap heavily between the two classes. Recall on *potable* is the weak spot.

## Possible improvements

- Gradient boosting (XGBoost / LightGBM) with tuned class weights
- Tune the decision threshold for the use case: when calling unsafe water "safe" is the costly mistake, favour precision on *potable*
- Cross-validated hyperparameter search instead of a single split

> Educational project. Not a substitute for laboratory water testing.
