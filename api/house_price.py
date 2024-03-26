from flask import Flask, request, jsonify
from flask import Blueprint
from flask_restful import Api, Resource
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

house_price_api = Blueprint('house_price_api', __name__, url_prefix='/api/house_price')
api = Api(house_price_api)

class HousePriceAPI(Resource):
    def __init__(self):
        # Load the dataset
        data = pd.read_csv('house_prices.csv')
        
        # Preprocessing and feature selection/engineering
        # Remove the 'furnishingstatus' column
        X = data.drop(['price', 'furnishingstatus'], axis=1)
        y = data['price']
        
        # Train the model
        self.model = LinearRegression()
        self.model.fit(X, y)

    def preprocess_inputs(self, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea):
        # Convert 'yes' or 'no' inputs to 1 or 0
        mainroad = 1 if mainroad.lower() == 'yes' else 0
        guestroom = 1 if guestroom.lower() == 'yes' else 0
        basement = 1 if basement.lower() == 'yes' else 0
        hotwaterheating = 1 if hotwaterheating.lower() == 'yes' else 0
        airconditioning = 1 if airconditioning.lower() == 'yes' else 0
        parking = 1 if parking.lower() == 'yes' else 0
        prefarea = 1 if prefarea.lower() == 'yes' else 0
        
        return area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea

    def predict_house_price(self, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea):
        # Prepare input data
        input_data = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'stories': [stories],
            'mainroad': [mainroad],
            'guestroom': [guestroom],
            'basement': [basement],
            'hotwaterheating': [hotwaterheating],
            'airconditioning': [airconditioning],
            'parking': [parking],
            'prefarea': [prefarea]
        })
        
        # Make prediction
        predicted_price = self.model.predict(input_data)
        return predicted_price[0]

    def post(self):
        try:
            # Get data from request
            data = request.json
            # Extract features
            area = data['area']
            bedrooms = data['bedrooms']
            bathrooms = data['bathrooms']
            stories = data['stories']
            mainroad = data['mainroad']
            guestroom = data['guestroom']
            basement = data['basement']
            hotwaterheating = data['hotwaterheating']
            airconditioning = data['airconditioning']
            parking = data['parking']
            prefarea = data['prefarea']
            # Preprocess inputs
            area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea = self.preprocess_inputs(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea)
            # Predict house price
            predicted_price = self.predict_house_price(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea)
            return jsonify({'predicted_price': predicted_price/10})
        except Exception as e:
            return jsonify({'error': str(e)})

api.add_resource(HousePriceAPI, '/predict')


"""
# Load the dataset
data = pd.read_csv('house_prices.csv')

# Preprocessing and feature selection/engineering
# Remove the 'furnishingstatus' column
X = data.drop(['price', 'furnishingstatus'], axis=1)
y = data['price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Define a function to make predictions
def predict_house_price(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea):
    # Prepare input data
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea]
    })
    
    # Make prediction
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
        predicted_price = model.predict(input_data)
    
    return predicted_price[0]

area = float(input("Enter the area of the house (in sqft): "))
bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of bathrooms: "))
stories = int(input("Enter the number of stories: "))
mainroad = int(input("Is the house located on the main road? (1 for Yes, 0 for No): "))
guestroom = int(input("Does the house have a guest room? (1 for Yes, 0 for No): "))
basement = int(input("Does the house have a basement? (1 for Yes, 0 for No): "))
hotwaterheating = int(input("Does the house have hot water heating? (1 for Yes, 0 for No): "))
airconditioning = int(input("Does the house have air conditioning? (1 for Yes, 0 for No): "))
parking = int(input("Enter the number of parking spaces: "))
prefarea = int(input("Is the house in a preferred area? (1 for Yes, 0 for No): "))

predicted_price = predict_house_price(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea)
print("Predicted price:", predicted_price/10)
"""