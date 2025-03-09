import numpy as np
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['target'])

st.title('Iris Classification App')
st.write('This app predicts the species of iris flower based on user input features')

st.sidebar.title('Input features')

sepal_length = st.sidebar.slider('Sepal length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider('Sepal width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider('Petal length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider('Petal width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=df.columns[:-1])


prediction = model.predict(input_features)
target_species = target_names[prediction][0]


st.markdown(
    f"<h3 style='color: blue; display: inline'>The predicted species is :</h3> "
    f"<h2 style='color: green;'> <strong>{target_species}</strong></h2>",
    unsafe_allow_html=True
)
