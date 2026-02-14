import streamlit as st
import joblib
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

st.write("Hello World")  # Works inside Streamlit app
print("Hello World")     # Works in terminal, but might not show in Streamlit UI

pipeline = joblib.load("/mount/src/ml-experiments/01_ClassificationModels/01_logisticreg_pipeline_1.pkl")


st.title("Heart Disease Prediction App")
