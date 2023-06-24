import requests
import streamlit as st
import pickle as pk
import sklearn
import numpy as np
import pandas as pd
import json
from streamlit_lottie import st_lottie

st.title('Bike Sharing Demand')
#load image
def load_lottieurl(url:str):
       r = requests.get(url)
       if r.status_code != 200:
           return None
       return r.json()
lottieurl_image = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_1s5rotap.json')

st_lottie(lottieurl_image,key="hello")
#--------------------------------------------

#load models
loaded_model= pk.load(open('C:/Users/AcTivE/Downloads/trained model.sav','rb'))
loaded_sc= pk.load(open('C:/Users/AcTivE/Downloads/trained Scaler.sav','rb'))

#prediction function
def prediction_fun(holiday, workingday, temp, humidity, rent_count, Hour, season_dummies, Weather_dummies, Day_of_week_dummies, month):
    rent_count = int(rent_count)
    month = int(month)
    seasons = [
        'Spring', 'Summer', 'Winter'
    ]
    season_du = [0 for i in range(3)]
    try:
        sea_id = seasons.index(season_dummies)
        season_du[sea_id] = 1
    except:
        pass
    # ---------------------
    Day_of_weeks = ['_Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
    Day_of_week_du = [0 for i in range(6)]
    try:
        dayw_id = Day_of_weeks.index(Day_of_week_dummies)
        Day_of_week_du[dayw_id] = 1
    except:
        pass
    # ---------------------
    wearthers =['Mist', 'Rainy', 'Snowy']
    Weather_du = [0 for i in range(3)]
    try:
        wea_id = wearthers.index(Weather_dummies)
        Weather_du[wea_id] = 1
    except:
        pass
    # ---------------------
    results = []
    lst=[holiday,workingday,temp,humidity,rent_count,Hour,season_du,Weather_du,Day_of_week_du,month ]
    def flatten(lst):
        if lst:
            car, *cdr = lst
            if isinstance(car, (list, tuple)):
                if cdr: return flatten(car) + flatten(cdr)
                return flatten(car)
            if cdr: return [car] + flatten(cdr)
            return [car]

    sc_aler = loaded_sc.transform(np.reshape(flatten(lst),(1, 19)))
    prediction = loaded_model.predict(sc_aler)
    return prediction

#------------------------------
holiday = st.radio(
                "holiday",
                  (1, 0, ))
workingday = st.radio(
                      "workingday",
                         (1, 0, ))

temp = st.slider('Temp',-30,50,10)

humidity = st.number_input('Humidity')

rent_count = st.number_input('Rent count')
Hour = st.number_input('Hour')

season_dummies = st.selectbox(
                         'season ',
                         ['Spring', 'Summer', 'Fall', 'Winter'])
Weather_dummies = st.selectbox(
                     'Weather',
                    ['Snowy', 'Rainy', 'Mist', 'Clear'])

Day_of_week_dummies = st.selectbox(
                         'Day of week ',
                     ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])

month = st.number_input('Month')

if st.button('Predict profit'):
    profit = prediction_fun(holiday, workingday, temp, humidity, rent_count, Hour, season_dummies, Weather_dummies, Day_of_week_dummies, month)
    st.success(f'The predicted profit of the Bike Sharing Demand is ${profit[0]:.2f} USD')