import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import joblib
from PIL import Image  # For handling images

st.markdown(
    """
    <style>
    /* Change the background color of the main content */
    .stApp {
        background-color: #F3E5F5; /* Violet color */
    }
  

    </style>
    """,
    unsafe_allow_html=True
)




st.title("Iris Flower Classifier")

sepal_length = st.number_input(
    label="Enter Sepal Length(cm):",
    min_value=4.3,    # Minimum value as a float
    max_value=7.9,  # Maximum value as a float
    value=4.3,       # Default value
    step=0.1          # Step size as a float
)
# Display the entered value
st.write(f"Sepal Length(cm): {sepal_length}")
st.markdown("<hr>", unsafe_allow_html=True)

sepal_width = st.number_input(
    label="Enter Sepal Width(cm):",
    min_value=2.0,    # Minimum value as a float
    max_value=4.4,  # Maximum value as a float
    value=2.0,       # Default value
    step=0.1          # Step size as a float
)
# Display the entered value
st.write(f"Sepal Width(cm): {sepal_width}")
st.markdown("<hr>", unsafe_allow_html=True)

petal_length = st.number_input(
    label="Enter Petal Length(cm):",
    min_value=1.0,    # Minimum value as a float
    max_value=6.9,  # Maximum value as a float
    value=1.0,       # Default value
    step=0.1          # Step size as a float
)
# Display the entered value
st.write(f"Petal Length(cm): {petal_length}")
st.markdown("<hr>", unsafe_allow_html=True)

petal_width = st.number_input(
    label="Enter Petal Width(cm):",
    min_value=0.1,    # Minimum value as a float
    max_value=2.5,  # Maximum value as a float
    value=0.1,       # Default value
    step=0.1          # Step size as a float
)
# Display the entered value
st.write(f"Petal Width(cm): {petal_width}")
st.markdown("<hr>", unsafe_allow_html=True)

loaded_model = joblib.load('Irismodel.pkl')
t = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = loaded_model.predict(t)



# Display the result and the corresponding image
if prediction[0] == "Iris-setosa":
    st.success("The predicted species is: Setosa")
    setosa_image = Image.open("Images/setosa.jpg")  # Replace with the path to your Setosa image
    st.image(setosa_image, caption="Iris Setosa", use_container_width=True)
elif prediction[0] == "Iris-versicolor":
    st.success("The predicted species is: Versicolor")
    versicolor_image = Image.open("Images/versicolor.jpg")  # Replace with the path to your Versicolor image
    st.image(versicolor_image, caption="Iris Versicolor", use_container_width=True)
elif prediction[0] == "Iris-virginica":
    st.success("The predicted species is: Virginica")
    virginica_image = Image.open("Images/virginica.jpg")  # Replace with the path to your Virginica image
    st.image(virginica_image, caption="Iris Virginica", use_container_width=True)
