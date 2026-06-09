from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="Churn Prediction Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
:root {
    --bg: #f5f7fb;
    --panel: #ffffff;
    --ink: #14213d;
    --muted: #64748b;
    --line: #dde4f0;
    --teal: #0f766e;
    --teal-soft: #dff6f1;
    --coral: #f97316;
    --coral-soft: #fff1e8;
    --violet: #6d5dfc;
    --mint: #14b8a6;
    --danger: #dc2626;
    --success: #16a34a;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(20, 184, 166, 0.14), transparent 34rem),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 48%, #eef4f8 100%);
    color: var(--ink);
}

section[data-testid="stSidebar"] {
    background: #0d1b2a;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] * {
    color: #edf6f9;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.14);
}

.main .block-container {
    padding-top: 1.25rem;
    max-width: 1280px;
}

.app-shell {
    display: grid;
    gap: 1rem;
}

.hero {
    background: linear-gradient(135deg, #ffffff 0%, #ecfeff 54%, #fff7ed 100%);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
}

.hero-row {
    align-items: center;
    display: flex;
    justify-content: space-between;
    gap: 1rem;
}

.hero h1 {
    color: var(--ink);
    font-size: clamp(1.9rem, 3vw, 3rem);
    font-weight: 800;
    letter-spacing: 0;
    margin: 0;
}

.hero p {
    color: #475569;
    font-size: 1rem;
    margin: .25rem 0 0;
    max-width: 58rem;
}

.status-pill {
    background: #0f766e;
    border-radius: 999px;
    color: white;
    display: inline-flex;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: 0;
    padding: .45rem .75rem;
    white-space: nowrap;
}

.metric-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    min-height: 118px;
    padding: 1rem;
}

.metric-card .label {
    color: var(--muted);
    font-size: .82rem;
    font-weight: 700;
    text-transform: uppercase;
}

.metric-card .value {
    color: var(--ink);
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
    margin-top: .45rem;
}

.metric-card .caption {
    color: var(--muted);
    font-size: .86rem;
    margin-top: .45rem;
}

.metric-card.teal { border-top: 4px solid var(--teal); }
.metric-card.coral { border-top: 4px solid var(--coral); }
.metric-card.violet { border-top: 4px solid var(--violet); }
.metric-card.mint { border-top: 4px solid var(--mint); }

.section-title {
    color: var(--ink);
    font-size: 1.2rem;
    font-weight: 800;
    margin: .25rem 0 .7rem;
}

.insight {
    background: #ffffff;
    border: 1px solid var(--line);
    border-left: 5px solid var(--teal);
    border-radius: 8px;
    padding: .85rem 1rem;
}

.insight strong { color: var(--ink); }
.insight span { color: var(--muted); }

.risk-high {
    background: #fee2e2;
    border: 1px solid #fecaca;
    color: #991b1b;
}

.risk-medium {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #9a3412;
}

.risk-low {
    background: #dcfce7;
    border: 1px solid #bbf7d0;
    color: #166534;
}

.risk-box {
    border-radius: 8px;
    font-size: 1.05rem;
    font-weight: 800;
    margin-top: .75rem;
    padding: .85rem 1rem;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: .8rem .9rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: .35rem;
}

.stTabs [data-baseweb="tab"] {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 8px 8px 0 0;
    color: var(--muted);
    font-weight: 700;
    padding: .65rem .9rem;
}

.stTabs [aria-selected="true"] {
    background: #0f766e !important;
    color: #ffffff !important;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 8px;
    border: 1px solid #0f766e;
    font-weight: 800;
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    background: #0f766e;
    border-color: #0f766e;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
}

@media (max-width: 820px) {
    .hero-row {
        align-items: flex-start;
        flex-direction: column;
    }

    .hero h1 {
        font-size: 2rem;
    }
}
</style>
"""


POSITIVE_LABEL_HINTS = {
    "yes",
    "y",
    "true",
    "1",
    "churn",
    "churned",
    "attrited customer",
    "exited",
    "lost",
}


TELCO_DEFAULTS: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 18,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.5,
    "TotalCharges": 1431.0,
}


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_columns: list[str]
    target_column: str
    positive_class: Any
    classes: np.ndarray
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba_positive: np.ndarray | None
    train_rows: int
    test_rows: int
    model_name: str


def inject_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def make_demo_data(rows: int = 1400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=rows,
        p=[0.56, 0.24, 0.20],
    )
    internet = rng.choice(
        ["Fiber optic", "DSL", "No"],
        size=rows,
        p=[0.45, 0.38, 0.17],
    )
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=rows,
        p=[0.35, 0.18, 0.24, 0.23],
    )
    tenure = rng.integers(1, 73, size=rows)
    monthly = np.round(
        rng.normal(64, 18, size=rows)
        + np.where(internet == "Fiber optic", 18, 0)
        + np.where(contract == "Two year", -9, 0),
        2,
    )
    monthly = np.clip(monthly, 18.5, 118.0)
    total = np.round(monthly * tenure + rng.normal(0, 95, size=rows), 2)
    total = np.clip(total, 20, None)

    security = rng.choice(["Yes", "No", "No internet service"], size=rows, p=[0.36, 0.48, 0.16])
    support = rng.choice(["Yes", "No", "No internet service"], size=rows, p=[0.34, 0.50, 0.16])
    partner = rng.choice(["Yes", "No"], size=rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=rows, p=[0.31, 0.69])
    paperless = rng.choice(["Yes", "No"], size=rows, p=[0.62, 0.38])
    senior = rng.choice([0, 1], size=rows, p=[0.84, 0.16])

    risk = (
        -1.95
        + (contract == "Month-to-month") * 1.05
        + (internet == "Fiber optic") * 0.62
        + (payment == "Electronic check") * 0.45
        + (security == "No") * 0.38
        + (support == "No") * 0.44
        + (paperless == "Yes") * 0.22
        + senior * 0.24
        - tenure * 0.032
        + (monthly - 65) * 0.012
        - (partner == "Yes") * 0.16
        - (dependents == "Yes") * 0.18
    )
    probability = 1 / (1 + np.exp(-risk))
    churn = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], size=rows),
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], size=rows, p=[0.91, 0.09]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=rows, p=[0.43, 0.48, 0.09]),
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=rows, p=[0.42, 0.42, 0.16]),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=rows, p=[0.41, 0.43, 0.16]),
            "TechSupport": support,
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=rows, p=[0.39, 0.45, 0.16]),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=rows, p=[0.40, 0.44, 0.16]),
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            as_text = cleaned[column].astype("string").str.strip()
            numeric = pd.to_numeric(as_text, errors="coerce")
            enough_numeric = numeric.notna().mean() >= 0.82
            if enough_numeric and as_text.nunique(dropna=True) > 8:
                cleaned[column] = numeric
            else:
                cleaned[column] = as_text

    return cleaned


def readable_feature(name: str) -> str:
    cleaned = name
    for prefix in ("num__", "cat__"):
        cleaned = cleaned.replace(prefix, "")
    return cleaned.replace("_", " ")


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def find_default_target(df: pd.DataFrame) -> str:
    preferred = ["Churn", "churn", "Exited", "Attrition", "target", "Target"]
    for column in preferred:
        if column in df.columns:
            return column
    return df.columns[-1]


def guess_positive_class(classes: np.ndarray) -> Any:
    for value in classes:
        if str(value).strip().casefold() in POSITIVE_LABEL_HINTS:
            return value
    if len(classes) == 2:
        return classes[1]
    return classes[0]


def load_uploaded_csv(uploaded_file: Any) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None

    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def split_feature_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str], list[str]]:
    features = [column for column in df.columns if column != target]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in features if column not in numeric]
    return features, numeric, categorical


@st.cache_resource(show_spinner=False)
def train_cached_model(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    test_size: float,
    random_state: int,
) -> ModelBundle:
    modeling_df = clean_dataframe(df).dropna(subset=[target]).copy()
    y = modeling_df[target]
    unique_classes = pd.Series(y).dropna().unique()

    if len(unique_classes) != 2:
        raise ValueError("La colonne cible doit contenir exactement deux classes.")

    features, numeric, categorical = split_feature_types(modeling_df, target)
    if not features:
        raise ValueError("Aucune colonne explicative disponible pour entrainer le modele.")

    x = modeling_df[features]
    stratify = y if y.value_counts().min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_ohe()),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    if model_name == "Random forest":
        classifier = RandomForestClassifier(
            n_estimators=320,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    classes = pipeline.named_steps["classifier"].classes_
    positive_class = guess_positive_class(classes)
    proba_positive = None
    if hasattr(pipeline, "predict_proba"):
        positive_index = list(classes).index(positive_class)
        proba_positive = pipeline.predict_proba(x_test)[:, positive_index]

    return ModelBundle(
        pipeline=pipeline,
        feature_columns=features,
        target_column=target,
        positive_class=positive_class,
        classes=classes,
        y_test=y_test,
        y_pred=y_pred,
        y_proba_positive=proba_positive,
        train_rows=len(x_train),
        test_rows=len(x_test),
        model_name=model_name,
    )


def render_header(rows: int, target: str, positive_class: Any) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-row">
                <div>
                    <h1>Churn Prediction Studio</h1>
                    <p>Tableau de bord operationnel pour analyser le churn telecom, entrainer un modele et scorer des clients.</p>
                </div>
                <span class="status-pill">{rows:,} lignes | cible {target}={positive_class}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str, tone: str = "teal") -> None:
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value * 100:.1f}%"


def churn_rate(df: pd.DataFrame, target: str, positive_class: Any) -> float:
    return (df[target].astype(str) == str(positive_class)).mean()


def plot_binary_distribution(df: pd.DataFrame, target: str, positive_class: Any) -> go.Figure:
    counts = df[target].astype(str).value_counts().reset_index()
    counts.columns = [target, "Clients"]
    colors = ["#f97316" if value == str(positive_class) else "#0f766e" for value in counts[target]]
    figure = go.Figure(
        data=[
            go.Pie(
                labels=counts[target],
                values=counts["Clients"],
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=3)),
                textinfo="percent+label",
            )
        ]
    )
    figure.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=35, b=10),
        title="Repartition de la cible",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#14213d"),
    )
    return figure


def plot_churn_by_category(df: pd.DataFrame, column: str, target: str, positive_class: Any) -> go.Figure:
    grouped = (
        df.assign(_positive=df[target].astype(str) == str(positive_class))
        .groupby(column, dropna=False)["_positive"]
        .mean()
        .reset_index()
        .sort_values("_positive", ascending=False)
    )
    grouped["_positive"] = grouped["_positive"] * 100
    grouped[column] = grouped[column].fillna("Missing").astype(str)

    figure = px.bar(
        grouped,
        x="_positive",
        y=column,
        orientation="h",
        labels={"_positive": "Taux de churn (%)", column: ""},
        color="_positive",
        color_continuous_scale=["#0f766e", "#f97316"],
    )
    figure.update_layout(
        height=max(320, 44 * len(grouped)),
        margin=dict(l=10, r=10, t=30, b=10),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=f"Churn par {column}",
        font=dict(color="#14213d"),
    )
    figure.update_xaxes(gridcolor="#e2e8f0")
    return figure


def plot_numeric_distribution(df: pd.DataFrame, column: str, target: str) -> go.Figure:
    figure = px.histogram(
        df,
        x=column,
        color=target,
        nbins=32,
        barmode="overlay",
        color_discrete_sequence=["#0f766e", "#f97316", "#6d5dfc"],
    )
    figure.update_traces(opacity=0.72)
    figure.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=35, b=10),
        title=f"Distribution de {column}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#14213d"),
        legend_title_text=target,
    )
    figure.update_yaxes(gridcolor="#e2e8f0")
    return figure


def render_dashboard(df: pd.DataFrame, target: str, bundle: ModelBundle) -> None:
    data = clean_dataframe(df).dropna(subset=[target]).copy()
    features, numeric, categorical = split_feature_types(data, target)

    rate = churn_rate(data, target, bundle.positive_class)
    avg_tenure = data["tenure"].mean() if "tenure" in data.columns else np.nan
    avg_monthly = data["MonthlyCharges"].mean() if "MonthlyCharges" in data.columns else np.nan

    cols = st.columns(4)
    with cols[0]:
        metric_card("Clients", f"{len(data):,}", "lignes exploitables", "teal")
    with cols[1]:
        metric_card("Taux de churn", format_percent(rate), f"classe positive: {bundle.positive_class}", "coral")
    with cols[2]:
        value = f"{avg_tenure:.1f}" if not pd.isna(avg_tenure) else "-"
        metric_card("Tenure moyen", value, "mois", "violet")
    with cols[3]:
        value = f"${avg_monthly:,.2f}" if not pd.isna(avg_monthly) else "-"
        metric_card("Revenu mensuel", value, "moyenne client", "mint")

    left, right = st.columns([0.95, 1.2])
    with left:
        st.plotly_chart(plot_binary_distribution(data, target, bundle.positive_class), width="stretch")
    with right:
        candidate = next((col for col in ["Contract", "InternetService", "PaymentMethod"] if col in categorical), None)
        if candidate:
            st.plotly_chart(
                plot_churn_by_category(data, candidate, target, bundle.positive_class),
                width="stretch",
            )
        elif categorical:
            st.plotly_chart(
                plot_churn_by_category(data, categorical[0], target, bundle.positive_class),
                width="stretch",
            )
        else:
            st.info("Aucune variable categorielle disponible.")

    chart_cols = st.columns(2)
    numeric_candidates = [col for col in ["tenure", "MonthlyCharges", "TotalCharges"] if col in numeric]
    for idx, column in enumerate(numeric_candidates[:2]):
        with chart_cols[idx]:
            st.plotly_chart(plot_numeric_distribution(data, column, target), width="stretch")

    st.markdown('<div class="section-title">Apercu des donnees</div>', unsafe_allow_html=True)
    st.dataframe(data.head(100), width="stretch", hide_index=True)


def model_metrics(bundle: ModelBundle) -> dict[str, float | None]:
    y_true_positive = bundle.y_test.astype(str) == str(bundle.positive_class)
    y_pred_positive = pd.Series(bundle.y_pred).astype(str) == str(bundle.positive_class)

    scores: dict[str, float | None] = {
        "Accuracy": accuracy_score(bundle.y_test, bundle.y_pred),
        "Precision": precision_score(y_true_positive, y_pred_positive, zero_division=0),
        "Recall": recall_score(y_true_positive, y_pred_positive, zero_division=0),
        "F1-score": f1_score(y_true_positive, y_pred_positive, zero_division=0),
        "ROC AUC": None,
    }

    if bundle.y_proba_positive is not None and len(np.unique(y_true_positive)) == 2:
        scores["ROC AUC"] = roc_auc_score(y_true_positive, bundle.y_proba_positive)
    return scores


def feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    classifier = bundle.pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        label = "importance"
    elif hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        label = "coefficient"
    else:
        return pd.DataFrame(columns=["feature", "impact", "abs_impact", "type"])

    result = pd.DataFrame(
        {
            "feature": [readable_feature(name) for name in names],
            "impact": values,
            "abs_impact": np.abs(values),
            "type": label,
        }
    )
    return result.sort_values("abs_impact", ascending=False)


def render_model(bundle: ModelBundle) -> None:
    scores = model_metrics(bundle)

    cols = st.columns(5)
    for index, (label, value) in enumerate(scores.items()):
        with cols[index]:
            st.metric(label, "-" if value is None else f"{value:.3f}")

    matrix = confusion_matrix(bundle.y_test.astype(str), pd.Series(bundle.y_pred).astype(str), labels=[str(c) for c in bundle.classes])
    matrix_df = pd.DataFrame(matrix, index=[f"Reel {c}" for c in bundle.classes], columns=[f"Pred {c}" for c in bundle.classes])

    left, right = st.columns([0.9, 1.1])
    with left:
        fig = px.imshow(
            matrix_df,
            text_auto=True,
            color_continuous_scale=["#ecfeff", "#0f766e"],
            aspect="auto",
        )
        fig.update_layout(
            title="Matrice de confusion",
            height=360,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#14213d"),
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        importance = feature_importance(bundle).head(14)
        if not importance.empty:
            fig = px.bar(
                importance.sort_values("abs_impact"),
                x="abs_impact",
                y="feature",
                orientation="h",
                color="impact",
                color_continuous_scale=["#0f766e", "#f8fafc", "#f97316"],
                labels={"abs_impact": "impact absolu", "feature": ""},
            )
            fig.update_layout(
                title="Variables les plus influentes",
                height=460,
                margin=dict(l=10, r=10, t=45, b=10),
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#14213d"),
            )
            fig.update_xaxes(gridcolor="#e2e8f0")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Importance des variables indisponible pour ce modele.")

    report = classification_report(bundle.y_test, bundle.y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "classe"})
    st.markdown('<div class="section-title">Rapport de classification</div>', unsafe_allow_html=True)
    st.dataframe(report_df, width="stretch", hide_index=True)


def widget_for_feature(df: pd.DataFrame, column: str, key_prefix: str) -> Any:
    series = df[column]
    default = TELCO_DEFAULTS.get(column)
    key = f"{key_prefix}_{column}"

    if pd.api.types.is_numeric_dtype(series):
        minimum = float(np.nanmin(series)) if series.notna().any() else 0.0
        maximum = float(np.nanmax(series)) if series.notna().any() else 100.0
        median = float(np.nanmedian(series)) if series.notna().any() else 0.0
        value = float(default) if default is not None else median
        value = min(max(value, minimum), maximum)
        non_null = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        has_only_whole_numbers = bool(len(non_null) and np.all(np.isclose(non_null % 1, 0)))
        step = 1.0 if has_only_whole_numbers else 0.1
        return st.number_input(column, min_value=minimum, max_value=maximum, value=value, step=step, key=key)

    options = sorted([str(value) for value in series.dropna().unique()])
    if not options:
        return st.text_input(column, value="" if default is None else str(default), key=key)

    default_text = str(default) if default is not None else options[0]
    index = options.index(default_text) if default_text in options else 0
    return st.selectbox(column, options=options, index=index, key=key)


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "Risque eleve", "risk-high"
    if probability >= 0.35:
        return "Risque moyen", "risk-medium"
    return "Risque faible", "risk-low"


def render_prediction(df: pd.DataFrame, bundle: ModelBundle) -> None:
    source = clean_dataframe(df)
    st.markdown('<div class="section-title">Prediction client</div>', unsafe_allow_html=True)

    input_values: dict[str, Any] = {}
    columns = st.columns(3)
    for index, feature in enumerate(bundle.feature_columns):
        with columns[index % 3]:
            input_values[feature] = widget_for_feature(source, feature, "single")

    input_df = pd.DataFrame([input_values], columns=bundle.feature_columns)

    if st.button("Calculer le risque", type="primary", width="stretch"):
        predicted = bundle.pipeline.predict(input_df)[0]
        if hasattr(bundle.pipeline, "predict_proba"):
            positive_index = list(bundle.classes).index(bundle.positive_class)
            probability = float(bundle.pipeline.predict_proba(input_df)[0, positive_index])
        else:
            probability = float(predicted == bundle.positive_class)

        label, css_class = risk_label(probability)
        st.markdown(
            f"""
            <div class="risk-box {css_class}">
                {label} - probabilite churn: {probability * 100:.1f}% | prediction: {predicted}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="insight">
                <strong>Action prioritaire</strong><br>
                <span>Segmenter ce client dans une campagne retention si le score depasse le seuil operationnel de votre equipe.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_batch_scoring(bundle: ModelBundle) -> None:
    st.markdown('<div class="section-title">Scoring batch</div>', unsafe_allow_html=True)
    scoring_file = st.file_uploader("CSV a scorer", type=["csv"], key="batch_csv")
    if scoring_file is None:
        st.info("Chargez un CSV contenant les memes variables explicatives que le modele.")
        return

    try:
        scoring = clean_dataframe(load_uploaded_csv(scoring_file))
        missing = [column for column in bundle.feature_columns if column not in scoring.columns]
        if missing:
            st.error(f"Colonnes manquantes: {', '.join(missing)}")
            return

        scored = scoring.copy()
        x_score = scored[bundle.feature_columns]
        scored["prediction_churn"] = bundle.pipeline.predict(x_score)
        if hasattr(bundle.pipeline, "predict_proba"):
            positive_index = list(bundle.classes).index(bundle.positive_class)
            scored["probabilite_churn"] = bundle.pipeline.predict_proba(x_score)[:, positive_index]
        st.dataframe(scored, width="stretch", hide_index=True)
        st.download_button(
            "Telecharger les scores",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="churn_scores.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )
    except Exception as exc:
        st.error(f"Scoring impossible: {exc}")


def render_sidebar(default_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, float, int]:
    with st.sidebar:
        st.title("Parametres")
        data_source = st.radio("Source", ["Dataset demo", "CSV"], horizontal=True)
        uploaded = None
        if data_source == "CSV":
            uploaded = st.file_uploader("Fichier CSV", type=["csv"], key="main_csv")

        if uploaded is not None:
            loaded = load_uploaded_csv(uploaded)
            if loaded is None:
                st.stop()
            df = clean_dataframe(loaded)
        else:
            df = default_df

        if df.empty:
            st.error("Le dataset est vide.")
            st.stop()

        default_target = find_default_target(df)
        target = st.selectbox(
            "Colonne cible",
            options=list(df.columns),
            index=list(df.columns).index(default_target),
        )
        model_name = st.selectbox("Modele", ["Logistic regression", "Random forest"], index=1)
        test_size = st.slider("Part test", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
        random_state = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)

    return df, target, model_name, test_size, int(random_state)


def main() -> None:
    inject_css()

    default_df = make_demo_data()
    df, target, model_name, test_size, random_state = render_sidebar(default_df)

    try:
        bundle = train_cached_model(df, target, model_name, float(test_size), int(random_state))
    except Exception as exc:
        st.error(f"Impossible d'entrainer le modele: {exc}")
        st.stop()

    render_header(len(df), target, bundle.positive_class)

    tabs = st.tabs(["Dashboard", "Modele", "Prediction", "Batch"])
    with tabs[0]:
        render_dashboard(df, target, bundle)
    with tabs[1]:
        render_model(bundle)
    with tabs[2]:
        render_prediction(df, bundle)
    with tabs[3]:
        render_batch_scoring(bundle)


if __name__ == "__main__":
    main()
