
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(page_title="Water Safety Predictor", page_icon="💧")

st.title("💧 Water Safety Predictor")
st.markdown("Enter water sample parameters to predict if it is safe to drink.")
st.markdown("---")

# Input sliders
col1, col2 = st.columns(2)

with col1:
    ph          = st.slider("pH level",           0.0, 14.0, 7.0, 0.1)
    hardness    = st.slider("Hardness (mg/L)",    47.0, 323.0, 150.0)
    solids      = st.slider("Solids (ppm)",       320.0, 61227.0, 20000.0)
    sulfate     = st.slider("Sulfate (mg/L)",     129.0, 481.0, 300.0)
    chloramines = st.slider("Chloramines (ppm)",  0.35, 13.12, 7.0, 0.1)

with col2:
    conductivity     = st.slider("Conductivity (μS/cm)", 181.0, 753.0, 400.0)
    organic_carbon   = st.slider("Organic Carbon (ppm)", 2.0, 28.0, 14.0, 0.1)
    trihalomethanes  = st.slider("Trihalomethanes (μg/L)", 0.0, 124.0, 66.0, 0.1)
    turbidity        = st.slider("Turbidity (NTU)",  1.45, 6.49, 4.0, 0.1)

st.markdown("---")

# Predict button
if st.button("🔍 Predict Water Safety"):
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.markdown("### Result")

    if prediction == 1:
        st.success(f"✅ SAFE TO DRINK — {probability[1]*100:.1f}% confidence")
    else:
        st.error(f"❌ NOT SAFE TO DRINK — {probability[0]*100:.1f}% confidence")

    # Show probability breakdown
    st.markdown("#### Confidence breakdown")
    col3, col4 = st.columns(2)
    col3.metric("Safe probability",     f"{probability[1]*100:.1f}%")
    col4.metric("Not safe probability", f"{probability[0]*100:.1f}%")

    # Connect back to original report
    st.markdown("---")
    st.markdown("#### 🔬 Microbial risk estimate (from your Poisson model)")
    import math
    lambda_est = turbidity * 0.45
    poisson_risk = 1 - math.exp(-lambda_est)
    st.info(f"Based on turbidity (NTU = {turbidity}), estimated λ = {lambda_est:.2f} "
            f"→ Probability of microbial presence: **{poisson_risk*100:.1f}%**")
    st.caption("This connects the ML prediction to the Poisson model from your original statistics report.")

st.markdown("---")
st.caption("Built with scikit-learn + Streamlit · Water Potability Dataset · BMSCE AI/ML")
