import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    roc_curve
)

from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Machine Learning Classification Project",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Machine Learning Classification Project")

st.write(
    """
    This project demonstrates four machine learning techniques
    using real-world datasets.

    **Models used:**
    - Random Forest
    - Logistic Regression
    - XGBoost
    - Decision Tree
    """
)

st.divider()


# =========================================================
# LOAD DIABETES DATA
# =========================================================

@st.cache_data
def load_diabetes_data():

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome"
    ]

    df = pd.read_csv(url, names=columns)

    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    df[zero_columns] = df[zero_columns].replace(0, np.nan)

    df.fillna(df.median(numeric_only=True), inplace=True)

    return df


# =========================================================
# LOAD TITANIC DATA
# =========================================================

@st.cache_data
def load_titanic_data():

    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

    df = pd.read_csv(url)

    df = df[
        [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Survived"
        ]
    ].copy()

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Sex"] = df["Sex"].map({
        "male": 0,
        "female": 1
    })

    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first=True
    )

    return df


# =========================================================
# PART 1 - RANDOM FOREST
# =========================================================

def random_forest_model():

    st.header("🌳 Part 1 — Random Forest: Breast Cancer")

    cancer = load_breast_cancer()

    X = pd.DataFrame(
        cancer.data,
        columns=cancer.feature_names
    )

    y = pd.Series(cancer.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")

    with col2:
        st.metric("ROC-AUC", f"{auc:.4f}")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": cancer.feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    ).head(10)

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.info(
        """
        **What this means:** Random Forest combines many decision trees.
        It predicts whether a breast-cancer sample belongs to one of the
        two classes. Feature importance shows which measurements were
        most useful to the model when making predictions.
        """
    )


# =========================================================
# PART 2 - LOGISTIC REGRESSION
# =========================================================

def logistic_regression_model():

    st.header("🩺 Part 2 — Logistic Regression: Diabetes")

    df = load_diabetes_data()

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:
        st.metric(
            "ROC-AUC",
            f"{auc:.4f}"
        )

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )

    st.subheader("Logistic Regression Coefficients")

    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    })

    coefficients = coefficients.sort_values(
        "Coefficient",
        ascending=False
    )

    st.dataframe(
        coefficients,
        use_container_width=True
    )

    st.info(
        """
        **What this means:** Logistic Regression estimates the probability
        that a person belongs to the diabetes class.

        A positive coefficient increases the predicted probability,
        while a negative coefficient decreases it, assuming the other
        variables remain fixed.
        """
    )


# =========================================================
# PART 3 - XGBOOST
# =========================================================

def xgboost_model():

    st.header("🚢 Part 3 — XGBoost: Titanic Survival")

    df = load_titanic_data()

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:
        st.metric(
            "ROC-AUC",
            f"{auc:.4f}"
        )

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.info(
        """
        **What this means:** XGBoost builds many small decision trees
        sequentially. Each new tree attempts to improve the mistakes
        made by previous trees.

        Feature importance shows which Titanic variables contributed
        most strongly to the model's predictions.
        """
    )


# =========================================================
# PART 4 - DECISION TREE
# =========================================================

def decision_tree_model():

    st.header("🌲 Part 4 — Decision Tree: Diabetes")

    df = load_diabetes_data()

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -------------------------------
    # FULL TREE
    # -------------------------------

    full_tree = DecisionTreeClassifier(
        random_state=42
    )

    full_tree.fit(
        X_train,
        y_train
    )

    full_predictions = full_tree.predict(
        X_test
    )

    full_accuracy = accuracy_score(
        y_test,
        full_predictions
    )

    # -------------------------------
    # RESTRICTED TREE
    # -------------------------------

    restricted_tree = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    restricted_tree.fit(
        X_train,
        y_train
    )

    restricted_predictions = restricted_tree.predict(
        X_test
    )

    restricted_accuracy = accuracy_score(
        y_test,
        restricted_predictions
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Full Tree Accuracy",
            f"{full_accuracy:.2%}"
        )

    with col2:
        st.metric(
            "Restricted Tree Accuracy",
            f"{restricted_accuracy:.2%}"
        )

    st.subheader("Accuracy Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Full Decision Tree",
            "Restricted Tree (max_depth=3)"
        ],
        "Accuracy": [
            full_accuracy,
            restricted_accuracy
        ]
    })

    st.bar_chart(
        comparison.set_index("Model")
    )

    st.subheader("Full Tree Classification Report")

    full_report = classification_report(
        y_test,
        full_predictions,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(full_report).transpose()
    )

    st.subheader("Restricted Tree Classification Report")

    restricted_report = classification_report(
        y_test,
        restricted_predictions,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(restricted_report).transpose()
    )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": full_tree.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.info(
        """
        **What this means:** The full decision tree is allowed to keep
        splitting the data until stopping conditions are reached.

        The restricted tree has max_depth=3, meaning it can only grow
        three levels deep.

        Restricting the tree can make it simpler and may reduce
        overfitting.
        """
    )


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📚 Project Navigation")

page = st.sidebar.radio(
    "Choose a section:",
    [
        "🏠 Home",
        "🌳 Random Forest",
        "🩺 Logistic Regression",
        "🚢 XGBoost",
        "🌲 Decision Tree"
    ]
)


# =========================================================
# PAGES
# =========================================================

if page == "🏠 Home":

    st.header("Welcome!")

    st.write(
        """
        This project demonstrates four classification algorithms
        using real-world datasets.

        Select a model from the menu on the left to explore its
        results.
        """
    )

    st.subheader("Datasets")

    st.markdown(
        """
        **1. Breast Cancer Dataset**

        Used with Random Forest to classify breast cancer samples.

        **2. Pima Indians Diabetes Dataset**

        Used with Logistic Regression and Decision Tree models
        to predict diabetes outcomes.

        **3. Titanic Dataset**

        Used with XGBoost to predict passenger survival.
        """
    )

    st.subheader("Important Evaluation Metrics")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC"
        ],
        "Meaning": [
            "Percentage of predictions that were correct.",
            "Of predicted positives, how many were actually positive.",
            "Of actual positives, how many were correctly detected.",
            "Balance between precision and recall.",
            "Measures how well the model separates the two classes."
        ]
    })

    st.table(metrics)


elif page == "🌳 Random Forest":

    random_forest_model()


elif page == "🩺 Logistic Regression":

    logistic_regression_model()


elif page == "🚢 XGBoost":

    xgboost_model()


elif page == "🌲 Decision Tree":

    decision_tree_model()