import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title("🌾 Crop Yield Prediction System")

st.write("Enter the values to predict crop yield")

# Sample dataset
data = {
    "Rainfall":[100,120,90,140,110,130],
    "Temperature":[25,27,24,29,26,28],
    "Fertilizer":[50,60,45,65,55,70],
    "Yield":[30,35,28,40,33,42]
}

df = pd.DataFrame(data)

# Training data
X = df[["Rainfall","Temperature","Fertilizer"]]
y = df["Yield"]

# Train model
model = LinearRegression()
model.fit(X,y)

# User inputs
st.subheader("Enter Agricultural Factors")

rainfall = st.number_input("Rainfall (mm)",0,300)
temperature = st.number_input("Temperature (°C)",0,50)
fertilizer = st.number_input("Fertilizer (kg)",0,100)

# Prediction button
if st.button("Predict Yield"):

    input_data = np.array([[rainfall,temperature,fertilizer]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Crop Yield: {prediction[0]:.2f} tons/hectare")

    # Graph
    st.subheader("Yield Graph")

    fig, ax = plt.subplots()
    ax.scatter(range(len(y)), y)
    ax.scatter(len(y), prediction, marker="x")
    ax.set_xlabel("Data Samples")
    ax.set_ylabel("Yield")
    ax.set_title("Crop Yield Prediction")

    st.pyplot(fig)

# Show dataset
st.subheader("Training Dataset")
st.write(df)