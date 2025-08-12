# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

DATA_PATH = "data/winequality-red.csv"
MODEL_PATH = "model.pkl"

@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=';')
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=',')
    except Exception:
        df = pd.read_csv(path, sep=',')
    return df

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

df = load_data()
model = load_model()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data", "Visualizations", "Predict", "Model"])

if page == "Home":
    # Custom background and creative layout
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(135deg, #f8e8ee 0%, #e0c3fc 100%);
        }
        .stApp {
            background: linear-gradient(135deg, #f8e8ee 0%, #e0c3fc 100%);
        }
        .wine-title {
            font-size: 3.5rem;
            font-weight: bold;
            color: #8e24aa;
            text-shadow: 2px 2px 8px #e1bee7;
            letter-spacing: 2px;
        }
        .wine-desc {
            font-size: 1.3rem;
            color: #4a148c;
            background: #f3e5f5;
            border-radius: 12px;
            padding: 1.2em;
            margin-bottom: 1.5em;
            box-shadow: 0 2px 8px #e1bee7;
        }
        .wine-icon {
            font-size: 2.5rem;
            margin-right: 0.5em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="wine-title">🍷 Wine Quality Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="wine-desc">Predict wine quality using machine learning regression models.<br>\
    <span style="color:#ad1457;font-weight:bold;">Instructions:</span> <ul><li>Place your dataset at <code>./data/dataset.csv</code></li><li>Run the training notebook/script to produce <code>model.pkl</code></li></ul></div>', unsafe_allow_html=True)
    st.markdown('<div style="display:flex;gap:2em;justify-content:center;align-items:center;margin-top:2em;">\
        <div style="background:#fce4ec;padding:1.5em 2em;border-radius:16px;box-shadow:0 2px 8px #f8bbd0;">\
            <span class="wine-icon">📊</span><b>Data Exploration</b><br><span style="color:#ad1457;">Visualize and understand your wine data</span>\
        </div>\
        <div style="background:#ede7f6;padding:1.5em 2em;border-radius:16px;box-shadow:0 2px 8px #b39ddb;">\
            <span class="wine-icon">🤖</span><b>Model Training</b><br><span style="color:#5e35b1;">Train and evaluate regression models</span>\
        </div>\
        <div style="background:#e8f5e9;padding:1.5em 2em;border-radius:16px;box-shadow:0 2px 8px #a5d6a7;">\
            <span class="wine-icon">🔮</span><b>Predict Quality</b><br><span style="color:#388e3c;">Get instant wine quality predictions</span>\
        </div>\
    </div>', unsafe_allow_html=True)

if page == "Data":
    st.header("Dataset")
    if df is None:
        st.error(f"Dataset not found at {DATA_PATH}. Place your CSV there (rename to dataset.csv).")
    else:
        st.write("Shape:", df.shape)
        st.dataframe(df.sample(15, random_state=42))
        if st.checkbox("Show summary stats"):
            st.write(df.describe().T)
        if st.checkbox("Show missing values"):
            st.write(df.isnull().sum())

if page == "Visualizations":
    st.header("Interactive Visualizations")
    if df is None:
        st.error("No dataset loaded.")
    else:
        st.subheader("Feature distribution")
        feature = st.selectbox("Select feature", [c for c in df.columns if c != "quality"])
        st.plotly_chart(px.histogram(df, x=feature, nbins=30, title=f"Distribution of {feature}"), use_container_width=True)

        st.subheader("Scatter (choose x and y)")
        x_col = st.selectbox("X", df.columns, index=list(df.columns).index("alcohol") if "alcohol" in df.columns else 0)
        y_col = st.selectbox("Y", df.columns, index=list(df.columns).index("quality") if "quality" in df.columns else 0)
        st.plotly_chart(px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"{y_col} vs {x_col}"), use_container_width=True)

        st.subheader("Correlation matrix")
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

if page == "Predict":
    st.header("Make a prediction")
    if df is None:
        st.error("Dataset not found. Place file at ./data/dataset.csv")
    elif model is None:
        st.error("Model not found. Run training to generate model.pkl in project root.")
    else:
        st.write("Enter inputs for features (sliders / numeric inputs).")
        inputs = {}
        for col in df.columns:
            if col == "quality": continue
            cmin, cmax = float(df[col].min()), float(df[col].max())
            cmedian = float(df[col].median())
            step = (cmax - cmin) / 100.0 if (cmax - cmin) > 0 else 0.01
            inputs[col] = st.number_input(col, min_value=cmin, max_value=cmax, value=cmedian, step=step, format="%.5f")

        if st.button("Predict"):
            X_new = pd.DataFrame([inputs])
            try:
                pred = model.predict(X_new)[0]
                st.success(f"Predicted quality: {pred:.3f}")
                # If ensemble, give optional ensemble std
                if hasattr(model, "estimators_"):
                    preds_each = np.array([est.predict(X_new)[0] for est in model.estimators_])
                    st.write("Ensemble std dev:", float(preds_each.std()))
            except Exception as e:
                st.error("Prediction failed: " + str(e))

if page == "Model":
    st.header("Model & performance")
    if os.path.exists("./test_metrics.txt"):
        st.markdown("**Test metrics (from training)**")
        with open("./test_metrics.txt", "r") as f:
            st.text(f.read())
    else:
        st.info("test_metrics.txt not found. Run training to create it.")
    if model is not None and df is not None:
        st.write("Quick holdout residuals (for a small check):")
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=["quality"])
        y = df["quality"]
        X_train, X_hold, y_train, y_hold = train_test_split(X,y,test_size=0.2,random_state=42)
        try:
            preds = model.predict(X_hold)
            res_df = pd.DataFrame({"actual": y_hold, "predicted": preds})
            st.plotly_chart(px.scatter(res_df, x="actual", y="predicted", title="Actual vs Predicted"), use_container_width=True)
            st.write("Residuals summary:", (res_df["actual"] - res_df["predicted"]).describe().to_frame())
        except Exception as e:
            st.error("Could not compute sample residuals: " + str(e))
