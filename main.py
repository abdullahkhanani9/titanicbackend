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
import threading

# import "packages" from flask
from flask import render_template,request  # import render_template from "public" flask libraries
from flask.cli import AppGroup


# import "packages" from "this" project
from __init__ import app, db, cors  # Definitions initialization


# setup APIs
from api.covid import covid_api # Blueprint import api definition
from api.joke import joke_api # Blueprint import api definition
from api.user import user_api # Blueprint import api definition
from api.player import player_api
# database migrations
from model.users import initUsers
from model.players import initPlayers

# setup App pages
from projects.projects import app_projects # Blueprint directory import projects definition


# Initialize the SQLAlchemy object to work with the Flask app instance
db.init_app(app)

# register URIs
app.register_blueprint(joke_api) # register api routes
app.register_blueprint(covid_api) # register api routes
app.register_blueprint(user_api) # register api routes
app.register_blueprint(player_api)
app.register_blueprint(app_projects) # register app pages

@app.errorhandler(404)  # catch for URL not found
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404

@app.route('/')  # connects default URL to index() function
def index():
    return render_template("index.html")

@app.route('/table/')  # connects /stub/ URL to stub() function
def table():
    return render_template("table.html")

@app.before_request
def before_request():
    # Check if the request came from a specific origin
    allowed_origin = request.headers.get('Origin')
    if allowed_origin in ['http://localhost:4100', 'http://127.0.0.1:4100', 'https://nighthawkcoders.github.io']:
        cors._origins = allowed_origin

# Create an AppGroup for custom commands
custom_cli = AppGroup('custom', help='Custom commands')

# Define a command to generate data
@custom_cli.command('generate_data')
def generate_data():
    initUsers()
    initPlayers()

# Register the custom command group with the Flask application
app.cli.add_command(custom_cli)
        
# this runs the application on the development server
if __name__ == "__main__":
    # change name for testing
    app.run(debug=True, host="0.0.0.0", port="8086")

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
