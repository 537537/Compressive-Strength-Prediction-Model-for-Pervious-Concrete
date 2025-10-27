import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ========== 页面配置 ==========
st.set_page_config(
    page_title="Pervious Concrete Strength Prediction",
    page_icon="🧱",
    layout="wide",
)

# ========== 自定义 CSS 样式 ==========
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e6f0ff, #ffffff);
}
h1 {
    text-align: center;
    color: #003366;
    font-family: 'Times New Roman', serif;
    font-weight: bold;
    font-size: 32px !important;
}
p, label {
    font-family: 'Times New Roman', serif;
    font-size: 16px;
}
div[data-baseweb="input"] > div {
    border-radius: 10px;
    border: 1px solid #80bfff;
}
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #007bff, #0056b3);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 0;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.3s ease;
    font-family: 'Times New Roman', serif;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #0056b3, #003d80);
    transform: scale(1.03);
}
.result-box {
    background-color: #f0f8ff;
    border: 2px solid #99ccff;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 64, 128, 0.1);
    margin-top: 25px;
}
.result-value {
    font-size: 26px;
    font-weight: bold;
    color: #003366;
}
</style>
""", unsafe_allow_html=True)

# ========== 页面标题 ==========
st.markdown("<h1>💧 Pervious Concrete Compressive Strength Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter the following 8 parameters to predict the compressive strength of pervious concrete (MPa).</p>", unsafe_allow_html=True)

# ========== 文件路径 ==========
MODEL_PATH = "final_catboost_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("⚠️ Model or scaler file is missing. Please check the file paths.")
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # ========== 输入界面 ==========
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            W_C = st.number_input("W/C (Water–Cement Ratio)", min_value=0.0, value=0.3, step=0.01)
            Dmin = st.number_input("Dmin (Minimum Aggregate Size)", min_value=0.0, value=4.75, step=0.01)
            Porosity = st.number_input("Porosity", min_value=0.0, value=15.0, step=0.1)
            Diameter = st.number_input("Size (Cylinder diameter / Cube side)", min_value=0.0, value=100.0, step=1.0)

        with col2:
            A_C = st.number_input("A/C (Aggregate–Cement Ratio)", min_value=0.0, value=3.0, step=0.1)
            ASR = st.number_input("ASR (Aggregate Size Ratio)", min_value=0.0, value=0.5, step=0.01)
            shape_option = st.selectbox("Specimen Shape", ["Cylinder", "Cube"])
            Shape = 1 if shape_option == "Cylinder" else 2
            Height = st.number_input("Specimen Height", min_value=0.0, value=200.0, step=1.0)

    # ========== 预测按钮 ==========
    predict_btn = st.button("🔮 Predict Compressive Strength")

    if predict_btn:
        try:
            # 输入组装与标准化
            input_data = np.array([[W_C, A_C, Dmin, ASR, Porosity, Shape, Diameter, Height]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            # 显示预测结果
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-value'>Predicted Compressive Strength: {prediction:.2f} MPa</div>
            </div>
            """, unsafe_allow_html=True)

            # ========== 特征重要性分析 ==========
            st.markdown("### 📊 Feature Importance Analysis")

            # 获取特征重要性
            try:
                importances = model.get_feature_importance()
            except Exception:
                st.warning("This model does not support feature importance extraction.")
                importances = None

            if importances is not None:
                feature_names = ["W/C", "A/C", "Dmin", "ASR", "Porosity", "Shape", "Diameter", "Height"]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(feature_names, importances, color="#007bff", alpha=0.8)
                ax.set_xlabel("Feature Importance (Weight)")
                ax.set_ylabel("Input Parameters")
                ax.set_title("Feature Influence on Prediction")
                ax.invert_yaxis()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {e}")

# ========== 底部提示 ==========
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:14px; color:gray;'>Developed by Q.D. | Powered by Streamlit & CatBoost</p>", unsafe_allow_html=True)
