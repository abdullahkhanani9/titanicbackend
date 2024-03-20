"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('house_prices.csv')

# Preprocessing and feature selection/engineering
# Split features (X) and target variable (y)
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a regression model (e.g., Linear Regression)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)


print(y_pred)
print(data.columns)
print(data.median())


print(data.query("airconditioning == 0").mean())

print("Script executed successfully")
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

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

# Example usage
predicted_price = predict_house_price(area=2800, bedrooms=4, bathrooms=3, stories=2, mainroad=1, guestroom=1, basement=0, hotwaterheating=1, airconditioning=1, parking=2, prefarea=1)
print("Predicted price:", predicted_price/10)
