"""
Step 2: Train and evaluate models.

We train TWO models on purpose:
1. Logistic Regression → simple, interpretable baseline
2. XGBoost → more powerful, less interpretable "real" model

Comparing them gives you a genuine interview talking point about the
interpretability vs. performance tradeoff.

Because our target is imbalanced (70% good / 30% risky), we:
- Use class_weight="balanced" (logistic regression) / scale_pos_weight (XGBoost)
  so the model doesn't just learn to predict "good" for everyone
- Evaluate with precision, recall, F1, and ROC-AUC — NOT accuracy alone,
  since accuracy is misleading on imbalanced data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
import joblib

from load_data import load_raw_data, clean_target


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(include="number").columns.tolist()

    return X, y, categorical_cols, numeric_cols


def build_preprocessor(categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    return preprocessor


def train_logistic_regression(preprocessor, X_train, y_train):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(preprocessor, X_train, y_train):
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
            )),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print(f"RESULTS: {model_name}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Good Credit", "Default Risk"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.3f}")
    print("\nConfusion Matrix:")
    print("                Predicted Good  Predicted Risk")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Actual Good        {cm[0][0]:>6}          {cm[0][1]:>6}")
    print(f"Actual Risk        {cm[1][0]:>6}          {cm[1][1]:>6}")


if __name__ == "__main__":
    df = load_raw_data()
    df = clean_target(df)

    X, y, categorical_cols, numeric_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    log_reg_model = train_logistic_regression(preprocessor, X_train, y_train)
    evaluate_model(log_reg_model, X_test, y_test, "Logistic Regression (baseline)")

    xgb_model = train_xgboost(preprocessor, X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    joblib.dump(xgb_model, "src/model.joblib")
    print("\n✅ Model saved to src/model.joblib")