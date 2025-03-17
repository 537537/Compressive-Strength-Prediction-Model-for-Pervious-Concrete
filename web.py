import streamlit as st
import numpy as np
import joblib
import os

# 自定义 CSS 以更改字体和按钮颜色
st.markdown(
    """
    <style>
    * { font-family: 'Times New Roman', serif; }
    
    div.stButton > button {
        background-color: #007BFF; /* 按钮背景颜色（蓝色） */
        color: white; /* 按钮文字颜色 */
        border-radius: 5px; /* 圆角边框 */
        border: 1px solid #0056b3; /* 按钮边框 */
        padding: 10px 20px;
        font-size: 16px;
    }
    
    div.stButton > button:hover {
        background-color: #0056b3; /* 鼠标悬停时变深蓝 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 设置 Streamlit 页面标题和描述
st.title("Pervious Concrete Compressive Strength Prediction Web Application")
st.markdown("Please enter the following 8 feature values, and the model will predict the compressive strength of pervious concrete (MPa).")

# 检查模型和标准化器文件是否存在
MODEL_PATH = "final_catboost_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler file is missing, please check the file path!")
else:
    # 加载模型和标准化器
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 使用 columns 创建两列布局
    col1, col2 = st.columns(2)

    with col1:
        W_C = st.number_input("W/C (Water-Cement Ratio)", min_value=0.0, value=0.0, step=0.01)
        Dmin = st.number_input("Dmin (Minimum Aggregate Size)", min_value=0.0, value=0.0, step=0.01)
        Porosity = st.number_input("Porosity", min_value=0.0, value=0.0, step=0.01)
        Diameter = st.number_input("Specimen Diameter (Edge Length for Cube)", min_value=0.0, value=0.0, step=0.1)

    with col2:
        A_C = st.number_input("A/C (Aggregate-Cement Ratio)", min_value=0.0, value=0.0, step=0.01)
        ASR = st.number_input("ASR (Aggregate Size Ratio)", min_value=0.0, value=0.0, step=0.01)
        shape_option = st.selectbox("Specimen Shape", ["Cylinder", "Cube"])
        Shape = 1 if shape_option == "Cylinder" else 2
        Height = st.number_input("Specimen Height", min_value=0.0, value=0.0, step=0.1)

    # 预测按钮
    if st.button("Predict Compressive Strength"):
        try:
            # 组装输入数据
            input_data = np.array([[W_C, A_C, Dmin, ASR, Porosity, Shape, Diameter, Height]])

            # 数据标准化
            input_scaled = scaler.transform(input_data)

            # 预测
            prediction = model.predict(input_scaled)[0]

            # 显示结果
            st.success(f"Predicted Compressive Strength of Pervious Concrete: {prediction:.2f} MPa")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# 运行方式提示
#st.markdown("**运行方式：** 请在终端中使用以下命令运行应用：")
#st.code("streamlit run web.py", language="sh")
