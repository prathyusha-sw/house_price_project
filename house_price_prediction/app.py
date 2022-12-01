import pickle
from pathlib import Path

import streamlit as st
import requests
import pandas as pd

MODEL_FILE = Path(__file__).parents[1] / 'notebooks' / 'model.pkl'
FEATURES = ['Sqr Ft', 'Beds', 'Bath', 'Years old', 'median_income', 'population']


def get_model(pickle_path):
    fin_mod = pickle.load(open(pickle_path, 'rb'))

    return fin_mod


def predict_house_price(sqrft, beds, bath, year_built, zipcode):
    # get median_income and population using zipcode
    url = f'https://api.census.gov/data/2020/acs/acs5?key=c08274ad3c650f2d11599a46a2474e284fad0c4d&get=B06011_001E&B01003_001E&for=zip%20code%20tabulation%20area:{zipcode}'
    resp = requests.get(url)
    median_income = int(resp.json()[1][0])
    population = int(resp.json()[1][1])

    # Calculate the age of the house as on 2020
    years_old = 2020 - year_built

    # Predict the house price
    x = pd.DataFrame([sqrft, beds, bath, years_old, median_income, population]).T

    x.columns = FEATURES
    fin_mod = get_model(MODEL_FILE)
    price = fin_mod.predict(x)

    return int(price[0])


# print(predict_house_price(1756, 3, 2, 1961, 75228))
st.set_page_config(layout="wide")
st.title('House Price Prediction')

sqrft = st.text_input("House area in sqrft",'3500')

beds = st.text_input("Number of bed rooms", '3')

bath = st.text_input("Number of bath rooms", '3.5')

year_built = st.text_input("Year built", '1998')

zipcode = st.text_input("zipcode", '19144')

inputs_str = [sqrft, beds, bath, year_built, zipcode]
if st.button("Submit"):
    inputs = [float(x) for x in inputs_str[:-1]]
    price = predict_house_price(*inputs, int(zipcode))

    st.markdown("""
    <style>
    .big-font {
        font-size:60px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f'<p class="big-font">Predicted price: ${price}</p>', unsafe_allow_html=True)



