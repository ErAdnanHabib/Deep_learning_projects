import pandas as pd
import pickle as pkl
import streamlit as st
from PIL import Image as img
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

pickle_in = open("classifier.pkl", "rb")
loaded_classifier = pkl.load(pickle_in)


column_names=[
    'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'
    ]

def prediction1(inputs):
    input_values = []
    for column in column_names:
            try:
                input_value = float(inputs[column])
                input_values.append(input_value)
            except ValueError:
                return "Invalid input. Please enter a valid number for {}".format(column)
    prediction = loaded_classifier.predict([input_values])
    return prediction


def main():
    st.markdown('<div style="background-color:#FFF00;padding:16px"><h1 style="color:#000000;text-align:center;">Diabetes Prediction</h1></div>',
                 unsafe_allow_html=True)
    
    user_inputs={}

    for column_name in column_names:
        user_input=st.text_input(f"Enter {column_name}")
        user_inputs[column_name]=user_input
    result=""

    if st.button("Predict"):
        result =prediction1(user_inputs)
    
    # Display the prediction result
    if result == [0]:
        st.success('PATIENT IS NOT DIABETIC :)')
    elif result == [1]:
        st.success('PATIENT IS DIABETIC !!!')
    else:
        st.success('MODEL IN PROCESS....')


if __name__=='__main__':
    main()
