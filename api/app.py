from flask import Flask, request, jsonify
import house_prices  # Import your model script

app = Flask(__name__)

# Define API endpoint
@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    # Get input data from request
    data = request.json
    
    # Extract input features
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
    
    # Make prediction using the model
    predicted_price = house_prices.predict_house_price(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea)
    
    # Return predicted price as JSON response
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
