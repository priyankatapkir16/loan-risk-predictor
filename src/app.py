"""
Step 4: Web interface (v3).

Same logic as v2, redesigned with a dark black/white theme and subtle
animations (fade-in on load, animated probability bar, hover transitions).
"""

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from load_data import load_raw_data, clean_target
from train_model import prepare_features, build_preprocessor, train_logistic_regression

app = Flask(__name__)

LABEL_MAPS = {
    "checking_account_status": {
        "A11": "Checking balance < 0 DM",
        "A12": "Checking balance 0-200 DM",
        "A13": "Checking balance >= 200 DM",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits taken / all paid back duly",
        "A31": "All credits at this bank paid back duly",
        "A32": "Existing credits paid back duly so far",
        "A33": "Delay in paying off in the past",
        "A34": "Critical account / credits at other banks",
    },
    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture / equipment",
        "A43": "Radio / television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Other",
    },
    "savings_account": {
        "A61": "Savings < 100 DM",
        "A62": "Savings 100-500 DM",
        "A63": "Savings 500-1000 DM",
        "A64": "Savings >= 1000 DM",
        "A65": "Unknown / no savings account",
    },
    "employment_since": {
        "A71": "Unemployed",
        "A72": "Employed < 1 year",
        "A73": "Employed 1-4 years",
        "A74": "Employed 4-7 years",
        "A75": "Employed >= 7 years",
    },
    "personal_status_sex": {
        "A91": "Male: divorced / separated",
        "A92": "Female: divorced / separated / married",
        "A93": "Male: single",
        "A94": "Male: married / widowed",
        "A95": "Female: single",
    },
    "other_debtors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property": {
        "A121": "Real estate",
        "A122": "Building society savings / life insurance",
        "A123": "Car or other property",
        "A124": "Unknown / no property",
    },
    "other_installment_plans": {
        "A141": "Other installment plans: bank",
        "A142": "Other installment plans: stores",
        "A143": "None",
    },
    "housing": {
        "A151": "Rents",
        "A152": "Owns home",
        "A153": "Lives for free",
    },
    "job": {
        "A171": "Unemployed / unskilled, non-resident",
        "A172": "Unskilled, resident",
        "A173": "Skilled employee / official",
        "A174": "Management / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "No",
        "A192": "Yes, registered",
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No",
    },
}

FRIENDLY_FIELD_NAMES = {
    "checking_account_status": "Checking account status",
    "duration_months": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Credit amount (DM)",
    "savings_account": "Savings account",
    "employment_since": "Employment duration",
    "installment_rate_pct": "Installment rate (% of income)",
    "personal_status_sex": "Personal status",
    "other_debtors": "Other debtors / guarantors",
    "residence_since": "Years at current residence",
    "property": "Property owned",
    "age": "Age",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "existing_credits_count": "Number of existing credits",
    "job": "Job type",
    "num_dependents": "Number of dependents",
    "telephone": "Has telephone",
    "foreign_worker": "Foreign worker",
}

df = load_raw_data()
df = clean_target(df)
X, y, categorical_cols, numeric_cols = prepare_features(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
preprocessor = build_preprocessor(categorical_cols, numeric_cols)
model = train_logistic_regression(preprocessor, X_train, y_train)

_classifier = model.named_steps["classifier"]
_prep = model.named_steps["preprocessor"]
_bg_transformed = _prep.transform(X_train.sample(n=50, random_state=42))
if hasattr(_bg_transformed, "toarray"):
    _bg_transformed = _bg_transformed.toarray()
_feature_names = _prep.get_feature_names_out()
_background_df = pd.DataFrame(_bg_transformed, columns=_feature_names)
_explainer = shap.Explainer(_classifier.predict_proba, _background_df)

CATEGORICAL_OPTIONS = {
    col: [(code, LABEL_MAPS[col].get(code, code)) for code in sorted(X[col].unique().tolist())]
    for col in categorical_cols
}
NUMERIC_RANGES = {
    col: (int(X[col].min()), int(X[col].max())) for col in numeric_cols
}


def readable_feature_name(raw_name: str) -> str:
    name = raw_name.replace("cat__", "").replace("num__", "")
    for col, friendly in FRIENDLY_FIELD_NAMES.items():
        if name.startswith(col + "_"):
            code = name[len(col) + 1:]
            label = LABEL_MAPS.get(col, {}).get(code, code)
            return f"{friendly}: {label}"
        if name == col:
            return friendly
    return raw_name


FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Risk Predictor</title>
    <style>
        * { box-sizing: border-box; }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        body {
            font-family: 'Segoe UI', -apple-system, Roboto, Arial, sans-serif;
            max-width: 820px;
            margin: 0 auto;
            padding: 56px 24px;
            background: #0a0a0a;
            color: #ededed;
        }
        .header {
            margin-bottom: 32px;
            animation: fadeInUp 0.5s ease both;
        }
        h1 {
            font-size: 27px;
            font-weight: 700;
            margin: 0 0 6px 0;
            letter-spacing: -0.5px;
            color: #fff;
        }
        .subtitle { color: #8a8a8a; font-size: 15px; margin: 0; }
        .card {
            background: #141414;
            border-radius: 14px;
            padding: 32px;
            border: 1px solid #262626;
            animation: fadeInUp 0.6s ease both;
            animation-delay: 0.1s;
        }
        .section-title {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6b6b6b;
            margin: 28px 0 14px 0;
        }
        .section-title:first-child { margin-top: 0; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
        .field { animation: fadeInUp 0.5s ease both; }
        .field label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: #b0b0b0;
            margin-bottom: 6px;
        }
        select, input[type=number] {
            width: 100%;
            padding: 10px 12px;
            border: 1.5px solid #2e2e2e;
            border-radius: 8px;
            font-size: 14px;
            color: #f2f2f2;
            background: #1c1c1c;
            transition: border-color 0.2s, background 0.2s, transform 0.1s;
        }
        select:hover, input:hover { border-color: #4a4a4a; }
        select:focus, input:focus {
            outline: none;
            border-color: #ffffff;
            background: #202020;
        }
        button {
            background: #ffffff;
            color: #0a0a0a;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 700;
            margin-top: 32px;
            width: 100%;
            transition: transform 0.15s ease, background 0.15s ease;
        }
        button:hover { background: #e2e2e2; transform: translateY(-1px); }
        button:active { transform: translateY(0); }
    </style>
</head>
<body>
    <div class="header">
        <h1>Loan Default Risk Predictor</h1>
        <p class="subtitle">Enter applicant details below to get a risk prediction with a plain-English explanation.</p>
    </div>
    <div class="card">
        <form method="POST" action="/predict">
            <div class="section-title">Financial Details</div>
            <div class="grid">
                {% for col in numeric_cols %}
                <div class="field">
                    <label>{{ friendly_names[col] }}</label>
                    <input type="number" name="{{ col }}" value="{{ ranges[col][0] }}" required>
                </div>
                {% endfor %}
            </div>
            <div class="section-title">Background</div>
            <div class="grid">
                {% for col in categorical_cols %}
                <div class="field">
                    <label>{{ friendly_names[col] }}</label>
                    <select name="{{ col }}">
                        {% for code, label in options[col] %}
                        <option value="{{ code }}">{{ label }}</option>
                        {% endfor %}
                    </select>
                </div>
                {% endfor %}
            </div>
            <button type="submit">Predict Risk</button>
        </form>
    </div>
</body>
</html>
"""

RESULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
    <style>
        * { box-sizing: border-box; }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes growBar {
            from { width: 0%; }
        }
        body {
            font-family: 'Segoe UI', -apple-system, Roboto, Arial, sans-serif;
            max-width: 820px;
            margin: 0 auto;
            padding: 56px 24px;
            background: #0a0a0a;
            color: #ededed;
        }
        h1 {
            font-size: 25px;
            font-weight: 700;
            margin-bottom: 24px;
            color: #fff;
            animation: fadeInUp 0.4s ease both;
        }
        .card {
            background: #141414;
            border-radius: 14px;
            padding: 32px;
            border: 1px solid #262626;
            animation: fadeInUp 0.5s ease both;
            animation-delay: 0.1s;
        }
        .result-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
        .badge-label { font-size: 18px; font-weight: 700; }
        .badge-label.risk { color: #ff6b6b; }
        .badge-label.safe { color: #51cf66; }
        .prob-text { color: #8a8a8a; font-size: 14px; margin-bottom: 10px; }
        .bar-track {
            width: 100%;
            height: 10px;
            background: #262626;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 28px;
        }
        .bar-fill {
            height: 100%;
            border-radius: 6px;
            animation: growBar 0.9s cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        .bar-fill.risk { background: linear-gradient(90deg, #ff6b6b, #fa5252); }
        .bar-fill.safe { background: linear-gradient(90deg, #51cf66, #40c057); }
        .section-title {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6b6b6b;
            margin-bottom: 14px;
        }
        .factor {
            padding: 14px 16px;
            margin-bottom: 10px;
            border-radius: 8px;
            font-size: 14px;
            background: #1a1a1a;
            border-left: 3px solid #444;
            animation: fadeInUp 0.5s ease both;
            transition: transform 0.15s ease, background 0.15s ease;
        }
        .factor:hover { transform: translateX(3px); background: #1f1f1f; }
        .factor.increases { border-left-color: #ff6b6b; }
        .factor.decreases { border-left-color: #51cf66; }
        .factor-name { font-weight: 600; color: #f2f2f2; }
        .factor-impact { color: #8a8a8a; font-size: 13px; margin-top: 2px; }
        a.back {
            display: inline-block;
            margin-top: 24px;
            color: #ffffff;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            border-bottom: 1px solid #444;
            transition: border-color 0.15s ease;
        }
        a.back:hover { border-color: #ffffff; }
    </style>
</head>
<body>
    <h1>Prediction Result</h1>
    <div class="card">
        <div class="result-header">
            <span class="badge-label {{ 'risk' if predicted_class == 'Default Risk' else 'safe' }}">
                {{ predicted_class }}
            </span>
        </div>
        <div class="prob-text">{{ probability }}</div>
        <div class="bar-track">
            <div class="bar-fill {{ 'risk' if predicted_class == 'Default Risk' else 'safe' }}" style="width: {{ prob_pct }}%;"></div>
        </div>
        <div class="section-title">Top Factors Influencing This Prediction</div>
        {% for f in factors %}
        <div class="factor {{ 'increases' if f.direction == 'increases risk' else 'decreases' }}" style="animation-delay: {{ loop.index0 * 0.05 }}s;">
            <div class="factor-name">{{ f.readable_feature }}</div>
            <div class="factor-impact">{{ f.direction | capitalize }} &middot; impact score {{ '%+.4f' % f.impact }}</div>
        </div>
        {% endfor %}
    </div>
    <a class="back" href="/">&larr; Try another applicant</a>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        FORM_TEMPLATE,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        options=CATEGORICAL_OPTIONS,
        ranges=NUMERIC_RANGES,
        friendly_names=FRIENDLY_FIELD_NAMES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    input_data = {}
    for col in numeric_cols:
        input_data[col] = [int(request.form[col])]
    for col in categorical_cols:
        input_data[col] = [request.form[col]]

    input_df = pd.DataFrame(input_data)[X.columns.tolist()]

    proba = model.predict_proba(input_df)[0]
    predicted_class = "Default Risk" if proba[1] > 0.5 else "Good Credit"
    risk_pct = proba[1] * 100

    transformed = _prep.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_df = pd.DataFrame(transformed, columns=_feature_names)

    shap_values = _explainer(transformed_df)
    contributions = shap_values.values[0, :, 1]

    contrib_df = pd.DataFrame({
        "feature": _feature_names,
        "value": transformed_df.iloc[0].values,
        "impact": contributions,
    })
    contrib_df["abs_impact"] = contrib_df["impact"].abs()
    contrib_df = contrib_df.sort_values("abs_impact", ascending=False).head(5)

    factors = [
        {
            "readable_feature": readable_feature_name(row["feature"]),
            "impact": row["impact"],
            "direction": "increases risk" if row["impact"] > 0 else "decreases risk",
        }
        for _, row in contrib_df.iterrows()
    ]

    # For the "Good Credit" case, show the bar as the confidence of being safe
    bar_pct = risk_pct if predicted_class == "Default Risk" else (100 - risk_pct)

    return render_template_string(
        RESULT_TEMPLATE,
        predicted_class=predicted_class,
        probability=f"{proba[1]:.1%} default risk probability",
        prob_pct=round(bar_pct, 1),
        factors=factors,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)