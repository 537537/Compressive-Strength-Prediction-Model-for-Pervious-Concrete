import streamlit as st
import numpy as np
import joblib
import os

# 设置 Streamlit 页面标题和描述
st.title("透水混凝土抗压强度预测 Web 应用")
st.markdown("请输入以下 8 个特征值，模型将预测透水混凝土的抗压强度 (MPa)")

# 检查模型和标准化器文件是否存在
MODEL_PATH = "final_catboost_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("模型或标准化器文件缺失，请检查文件路径！")
else:
    # 加载模型和标准化器
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 创建输入框（无上限，最小值设为0）
    W_C = st.number_input("W/C (水胶比)", min_value=0.0, value=0.5, step=0.01)
    A_C = st.number_input("A/C (骨料胶比)", min_value=0.0, value=3.0, step=0.1)
    Dmin = st.number_input("Dmin (最小粒径)", min_value=0.0, value=10.0, step=0.1)
    ASR = st.number_input("ASR (骨料粒径比)", min_value=0.0, value=0.1, step=0.01)
    Porosity = st.number_input("Porosity (孔隙率)", min_value=0.0, value=0.2, step=0.01)

    # 使用 selectbox 创建选择框
    shape_option = st.selectbox("Shape (试样形状)", ["圆柱体", "立方体"])
    Shape = 1 if shape_option == "圆柱体" else 2  # 圆柱体 -> 1，立方体 -> 2

    Diameter = st.number_input("Diameter (试样直径)", min_value=0.0, value=20.0, step=0.1)
    Height = st.number_input("Height (试样高度)", min_value=0.0, value=30.0, step=0.5)

    # 预测按钮
    if st.button("预测抗压强度"):
        try:
            # 组装输入数据
            input_data = np.array([[W_C, A_C, Dmin, ASR, Porosity, Shape, Diameter, Height]])

            # 数据标准化
            input_scaled = scaler.transform(input_data)

            # 预测
            prediction = model.predict(input_scaled)[0]

            # 显示结果
            st.success(f"预测的透水混凝土抗压强度：{prediction:.2f} MPa")
        except Exception as e:
            st.error(f"预测时发生错误: {e}")

# 运行方式提示
st.markdown("**运行方式：** 请在终端中使用以下命令运行应用：")
st.code("streamlit run web.py", language="sh")
