from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('Final_ET_Model')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


def run():
    from PIL import Image
    image = Image.open('logo.png')
    image_profile = Image.open('profile.jpg')

    st.image(image, use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))

    st.sidebar.info('This app is created to predict happiness of countries')
    st.sidebar.success('Created by: Muhammad Umer Sajjad')

    st.sidebar.image(image_hospital)

    st.title("Countries Happiness Prediction App")

    if add_selectbox == 'Online':

        GDP_per_Capita = st.number_input('Log of GDP_per_Capita', min_value=0.00, max_value=2.30, value=0.97)
        Social_Support = st.number_input('Annual Avg of Social_Support', min_value=0.00, max_value=1.74, value=1.03)
        Healthy_Life_Expectancy = st.number_input('Healthy Life Expectancy', min_value=0.00, max_value=1.24,
                                                  value=0.60)
        Freedom = st.number_input('Annual Avg of Freedom', min_value=0.00, max_value=0.84, value=0.44)
        Government_Corruption_Perception = st.number_input('Annual Avg of Government Corruption Perception', min_value=0.00,
                                                           max_value=0.68, value=0.13)
        Generosity = st.number_input('Annual Avg of Generosity', min_value=0.00, max_value=0.93, value=0.20)
        Regional_Indicator = st.selectbox('Region of Your Country', [west_europe,north_america,middle_east,latin_america,central_europe,east_asia,south_east_asia,commonwealth_independent,sub_saharan_africa,south_asia])

        #         if st.checkbox('Smoker'):
        #             smoker = 'yes'
        #         else:
        #             smoker = 'no'
        #         region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        #         output=""

        input_dict = {'GDP_per_Capita': GDP_per_Capita, 'Social_Support': Social_Support,
                      'Healthy_Life_Expectancy': Healthy_Life_Expectancy, 'Freedom': Freedom,
                      'Government_Corruption_Perception': Government_Corruption_Perception, 'Generosity': Generosity, 'Regional_Indicator': Regional_Indicator}

        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = 'The Happiness Score: ' + str(output)

            st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
