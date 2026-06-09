import streamlit as st
import pandas as pd

try:
    import joblib
except ImportError:
    st.error(
        "Missing dependency: joblib is not installed. "
        "Please add `joblib` to requirements.txt and redeploy."
    )
    st.stop()


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="🏦 Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.header-box {
    background: linear-gradient(90deg,#1e3c72,#2a5298);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

.result-success {
    padding:20px;
    border-radius:15px;
    background:#d4edda;
    color:#155724;
    font-size:24px;
    text-align:center;
    font-weight:bold;
}

.result-danger {
    padding:20px;
    border-radius:15px;
    background:#f8d7da;
    color:#721c24;
    font-size:24px;
    text-align:center;
    font-weight:bold;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#11998e,#38ef7d);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    height: 55px;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.02);
}

.footer {
    text-align:center;
    color:gray;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Load Model Files
# =========================
artifacts = joblib.load("model_prod_files.pkl")

model = artifacts.get("model")
scaler = (
    artifacts.get("scaler")
    or artifacts.get("num_encod")
    or artifacts.get("minmax_scaler")
)
ohe = (
    artifacts.get("ohe")
    or artifacts.get("cat_encod")
    or artifacts.get("onehotencoder")
    or artifacts.get("onehot_encoder")
)

if model is None or scaler is None or ohe is None:
    st.error(
        "Unable to load model artifacts. "
        "Please ensure 'model_prod_files.pkl' contains keys "
        "'model', 'scaler' (or 'num_encod'), and 'ohe' (or 'cat_encod')."
    )
    st.stop()

# =========================
# Sidebar
# =========================
with st.sidebar:

    st.title("📊 Project Details")

    st.info("""
    Model Used:
    KNeighborsClassifier

    Data Processing:
    • OneHotEncoder
    • MinMaxScaler

    Objective:
    Predict whether a customer
    will get loan approval.
    """)

# =========================
# Header
# =========================
st.markdown("""
<div class='header-box'>
<h1>🏦 Loan Approval Prediction System</h1>
<p>Machine Learning Powered Banking Solution</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Inputs
# =========================
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    job = st.selectbox(
        "Job",
        [
            "admin.",
            "technician",
            "services",
            "management",
            "retired",
            "blue-collar",
            "unemployed",
            "entrepreneur",
            "housemaid",
            "student",
            "self-employed",
            "unknown"
        ]
    )

    marital = st.selectbox(
        "Marital Status",
        ["single", "married", "divorced"]
    )

    education = st.selectbox(
        "Education",
        ["primary", "secondary", "tertiary", "unknown"]
    )

    default = st.selectbox(
        "Credit Default",
        ["yes", "no"]
    )

    balance = st.number_input(
        "Account Balance",
        value=1000
    )

    housing_loan = st.selectbox(
        "Housing Loan",
        ["yes", "no"]
    )

    personal_loan = st.selectbox(
        "Personal Loan",
        ["yes", "no"]
    )

with col2:

    contact = st.selectbox(
        "Contact Type",
        ["cellular", "telephone", "unknown"]
    )

    day = st.number_input(
        "Last Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )

    month = st.selectbox(
        "Month",
        [
            "jan", "feb", "mar", "apr",
            "may", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec"
        ]
    )

    duration = st.number_input(
        "Call Duration",
        min_value=0,
        value=300
    )

    campaign = st.number_input(
        "Campaign Contacts",
        min_value=0,
        value=1
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        value=999
    )

    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        value=0
    )

    poutcome = st.selectbox(
        "Previous Outcome",
        ["success", "failure", "other", "unknown"]
    )

# =========================
# Predict Button
# =========================
if st.button("🔍 Predict Loan Approval"):

    input_df = pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "balance": [balance],
        "housing_loan": [housing_loan],
        "personal_loan": [personal_loan],
        "contact": [contact],
        "day": [day],
        "month": [month],
        "duration": [duration],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome]
    })

    st.subheader("📋 Customer Details")
    st.dataframe(input_df, use_container_width=True)

    num_cols = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays"
    ]

    cat_cols = [
        "job",
        "marital",
        "education",
        "contact",
        "month",
        "poutcome"
    ]

    bin_cols = [
        "default",
        "housing_loan",
        "personal_loan"
    ]

    try:

        num_trans = pd.DataFrame(
            scaler.transform(input_df[num_cols].to_numpy()),
            columns=num_cols
        )

        cat_trans = ohe.transform(input_df[cat_cols].to_numpy())
        if hasattr(cat_trans, "toarray"):
            cat_trans = cat_trans.toarray()
        cat_trans = pd.DataFrame(
            cat_trans,
            columns=ohe.get_feature_names_out(cat_cols)
        )

        bin_trans = input_df[bin_cols].replace({"yes": 1, "no": 0})
        prev_trans = input_df[["previous"]]

        final_input = pd.concat(
            [
                num_trans.reset_index(drop=True),
                cat_trans.reset_index(drop=True),
                bin_trans.reset_index(drop=True),
                prev_trans.reset_index(drop=True)
            ],
            axis=1
        )

        prediction = model.predict(
            final_input
        )

        try:
            probability = model.predict_proba(
                final_input
            )

            confidence = probability.max() * 100

        except:
            confidence = None

        st.markdown("---")

        if str(prediction[0]).lower() in [
            "yes",
            "1",
            "approved",
            "approve"
        ]:

            st.markdown("""
            <div class='result-success'>
            ✅ LOAN APPROVED
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='result-danger'>
            ❌ LOAN REJECTED
            </div>
            """, unsafe_allow_html=True)

        if confidence:

            st.subheader(
                "📈 Prediction Confidence"
            )

            st.progress(
                int(confidence)
            )

            st.write(
                f"Confidence Score: {confidence:.2f}%"
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# =========================
# Footer
# =========================
st.markdown("---")

st.markdown("""
<div class='footer'>
Built with ❤️ using Streamlit, KNN, MinMaxScaler and OneHotEncoder
</div>
""", unsafe_allow_html=True)
