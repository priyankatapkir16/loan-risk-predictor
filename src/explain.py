"""
Step 3: Explainability with SHAP.

We use SHAP (SHapley Additive exPlanations) to answer: "WHY did the model
flag this specific applicant as risky?"

This is the difference between a model that just outputs a number, and a
tool that's actually usable in a regulated industry like lending, where
"the model said no" isn't a legally or ethically acceptable answer on its
own — you need to say WHY.

We explain the Logistic Regression model specifically, since Step 2 showed
it has better recall on catching actual defaults than XGBoost on this
dataset — that's our chosen "primary" model, not just the simpler one.

IMPORTANT DESIGN NOTE: SHAP needs numeric input. Our raw data has text
columns (like "checking_account_status" = "A11"), so we explain the model
AFTER preprocessing (one-hot encoding + scaling), not on the raw data.
This means feature names in the explanation look like
"checking_account_status_A11" (one column per category) rather than the
original clean column names — that's expected and correct.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from load_data import load_raw_data, clean_target
from train_model import prepare_features, build_preprocessor, train_logistic_regression
from sklearn.model_selection import train_test_split


def explain_single_prediction(pipeline, X_train, X_test, row_index: int = 0):
    """
    Explain ONE prediction in detail — this is what powers a real feature
    like "why was I flagged as risky" in a user-facing app.

    We pull the preprocessor and classifier out of the pipeline, transform
    the data to numeric form first, then run SHAP on the classifier alone.
    This sidesteps SHAP's masker choking on raw text columns.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    background = X_train_df.sample(n=min(50, len(X_train_df)), random_state=42)

    explainer = shap.Explainer(classifier.predict_proba, background)

    single_applicant = X_test_df.iloc[[row_index]]
    shap_values = explainer(single_applicant)

    prediction_proba = classifier.predict_proba(single_applicant.values)[0]
    predicted_class = "Default Risk" if prediction_proba[1] > 0.5 else "Good Credit"

    print("=" * 60)
    print(f"EXPLAINING APPLICANT #{row_index}")
    print("=" * 60)
    print(f"Predicted: {predicted_class}")
    print(f"Probability of default risk: {prediction_proba[1]:.2%}")
    print()

    contributions = shap_values.values[0, :, 1]  # class index 1 = Default Risk
    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "applicant_value": single_applicant.iloc[0].values,
        "contribution": contributions,
    })
    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    contrib_df = contrib_df.sort_values("abs_contribution", ascending=False)

    print("TOP 5 FACTORS INFLUENCING THIS PREDICTION:")
    print("(positive = pushes toward 'Default Risk', negative = pushes toward 'Good Credit')")
    print()
    for _, row in contrib_df.head(5).iterrows():
        direction = "increases risk" if row["contribution"] > 0 else "decreases risk"
        print(f"  {row['feature']:<40} = {row['applicant_value']:.2f}  "
              f"| impact: {row['contribution']:+.4f}  ({direction})")

    return contrib_df


def plot_global_importance(pipeline, X_train, sample_size: int = 100):
    """
    Beyond explaining ONE applicant, this shows which features matter
    MOST across many applicants overall — useful for a summary chart
    in your app or README.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_train_transformed = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    background = X_train_df.sample(n=min(50, len(X_train_df)), random_state=42)
    explanation_sample = X_train_df.sample(n=min(sample_size, len(X_train_df)), random_state=1)

    explainer = shap.Explainer(classifier.predict_proba, background)
    shap_values = explainer(explanation_sample)

    shap.summary_plot(
        shap_values[:, :, 1],
        explanation_sample,
        show=False,
    )
    plt.tight_layout()
    plt.savefig("src/shap_summary.png", dpi=150)
    plt.close()
    print("\n[SAVED] Global feature importance chart saved to src/shap_summary.png")


if __name__ == "__main__":
    df = load_raw_data()
    df = clean_target(df)

    X, y, categorical_cols, numeric_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    model = train_logistic_regression(preprocessor, X_train, y_train)

    explain_single_prediction(model, X_train, X_test, row_index=0)

    probas = model.predict_proba(X_test)[:, 1]
    highest_risk_index = int(np.argmax(probas))
    print("\n")
    explain_single_prediction(model, X_train, X_test, row_index=highest_risk_index)

    plot_global_importance(model, X_train)