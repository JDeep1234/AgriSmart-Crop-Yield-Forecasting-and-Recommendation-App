import pandas as pd
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import joblib
from sklearn.ensemble import ExtraTreesRegressor

def train_and_recommend(crop, season, state, area, production, annual_rainfall):

    # Map crop name to its corresponding integer value
    crop_id = [key for key, value in crop_map.items() if value == crop][0]

    # Map season name to its corresponding integer value
    season_id = [key for key, value in season_map.items() if value == season][0]

    # Map state name to its corresponding integer value
    state_id = [key for key, value in state_map.items() if value == state][0]

    df = pd.read_csv('crop_yield.csv')

    unique_values = df['Crop'].unique()
    for i, crop in enumerate(unique_values, 1):
        df['Crop'].replace(crop, i, inplace=True)
    unique_values1 = df['Season'].unique()
    for i, crop in enumerate(unique_values1, 1):
        df['Season'].replace(crop, i, inplace=True)
    unique_values2 = df['State'].unique()
    for i, crop in enumerate(unique_values2, 1):
        df['State'].replace(crop, i, inplace=True)

    X = df[['Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Crop']]
    y = df[['Fertilizer','Pesticide']]

    model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Prepare input features as a list or numpy array
    input_features = [[season_id, state_id, area, production, annual_rainfall, crop_id]]

    # Make prediction using the trained model
    prediction = model.predict(input_features)

    # Extract fertilizer and pesticide recommendations from the prediction
    max_yield_fertilizer, max_yield_pesticide = prediction[0]

    return max_yield_fertilizer, max_yield_pesticide


# Load the recommendation model
recommendation_model = joblib.load('trained_model.pkl')

# Dictionary mapping integer values to crop names
crop_map = {
    1: 'Arecanut', 2: 'Arhar/Tur', 3: 'Castor seed', 4: 'Coconut',
    5: 'Cotton(lint)', 6: 'Dry chillies', 7: 'Gram', 8: 'Jute', 9: 'Linseed',
    10: 'Maize', 11: 'Mesta', 12: 'Niger seed', 13: 'Onion', 14: 'Other Rabi pulses',
    15: 'Potato', 16: 'Rapeseed & Mustard', 17: 'Rice', 18: 'Sesamum',
    19: 'Small millets', 20: 'Sugarcane', 21: 'Sweet potato', 22: 'Tapioca',
    23: 'Tobacco', 24: 'Turmeric', 25: 'Wheat', 26: 'Bajra', 27: 'Black pepper',
    28: 'Cardamom', 29: 'Coriander', 30: 'Garlic', 31: 'Ginger', 32: 'Groundnut',
    33: 'Horse-gram', 34: 'Jowar', 35: 'Ragi', 36: 'Cashewnut', 37: 'Banana',
    38: 'Soyabean', 39: 'Barley', 40: 'Khesari', 41: 'Masoor', 42: 'Moong(Green Gram)',
    43: 'Other Kharif pulses', 44: 'Safflower', 45: 'Sannhamp', 46: 'Sunflower',
    47: 'Urad', 48: 'Peas & beans (Pulses)', 49: 'other oilseeds', 50: 'Other Cereals',
    51: 'Cowpea(Lobia)', 52: 'Oilseeds total', 53: 'Guar seed', 54: 'Other Summer Pulses',
    55: 'Moth'
}

# Dictionary mapping integer values to seasons
season_map = {
    1: 'Whole Year', 2: 'Kharif', 3: 'Rabi', 4: 'Autumn', 5: 'Summer', 6: 'Winter'
}

# Dictionary mapping integer values to states
state_map = {
    1: 'Assam', 2: 'Karnataka', 3: 'Kerala', 4: 'Meghalaya', 5: 'West Bengal', 
    6: 'Puducherry', 7: 'Goa', 8: 'Andhra Pradesh', 9: 'Tamil Nadu', 10: 'Odisha', 
    11: 'Bihar', 12: 'Gujarat', 13: 'Madhya Pradesh', 14: 'Maharashtra', 
    15: 'Mizoram', 16: 'Punjab', 17: 'Uttar Pradesh', 18: 'Haryana', 
    19: 'Himachal Pradesh', 20: 'Tripura', 21: 'Nagaland', 22: 'Chhattisgarh', 
    23: 'Uttarakhand', 24: 'Jharkhand', 25: 'Delhi', 26: 'Manipur', 
    27: 'Jammu and Kashmir', 28: 'Telangana', 29: 'Arunachal Pradesh', 
    30: 'Sikkim'
}

# Streamlit UI
st.title('AgriSmart: Crop Yield Forecasting and Recommendation App')

tabs = st.columns(2)
if tabs[0].button('Predict Crop Yield'):
    st.session_state.show_yield = True
    st.session_state.show_calculator = False
if tabs[1].button('Smart Fertilizer and Pesticide Calculator'):
    st.session_state.show_yield = False
    st.session_state.show_calculator = True

# Display the selected subpage
if 'show_yield' not in st.session_state:
    st.session_state.show_yield = True
if 'show_calculator' not in st.session_state:
    st.session_state.show_calculator = False

# Display image
image_url = "https://picsum.photos/800/400?category=nature"
st.image(image_url, use_column_width=True)

# Add buttons for Prediction and Recommendation


if st.session_state.show_yield:
    st.subheader("Prediction Inputs")
    crop = st.selectbox('Select Crop:', options=list(crop_map.values()), index=0)
    season = st.selectbox('Select Season:', options=list(season_map.values()), index=0)
    state = st.selectbox('Select State:', options=list(state_map.values()), index=0)
    area = st.number_input('Enter Area (in hectares):', value=0.0)
    production = st.number_input('Enter Production (in metric tons):', value=0.0)
    annual_rainfall = st.number_input('Enter Annual Rainfall (in mm):', value=0.0)
    fertilizer = st.number_input('Enter Fertilizer (in kg):', value=0.0)
    pesticide = st.number_input('Enter Pesticide (in kg):', value=0.0)

    if st.button('Predict'):
        prediction = train_and_recommend(crop, season, state, area, production, annual_rainfall)
        st.subheader('Yield Prediction:')
        st.success(f"The predicted yield is {prediction[0]} per unit area")

elif st.session_state.show_calculator:
    st.subheader("Recommendation Inputs")
    crop = st.selectbox('Select Crop:', options=list(crop_map.values()), index=0)
    season = st.selectbox('Select Season:', options=list(season_map.values()), index=0)
    state = st.selectbox('Select State:', options=list(state_map.values()), index=0)
    area = st.number_input('Enter Area (in hectares):', value=0.0)
    production = st.number_input('Enter Production (in metric tons):', value=0.0)
    annual_rainfall = st.number_input('Enter Annual Rainfall (in mm):', value=0.0)

    if st.button('Recommend'):
        fertilizer, pesticide = train_and_recommend(crop, season, state, area, production, annual_rainfall)
        st.subheader('Recommendation:')
        st.success(f"Recommended Fertilizer: {fertilizer} Kg")
        st.success(f"Recommended Pesticide: {pesticide} Kg")
