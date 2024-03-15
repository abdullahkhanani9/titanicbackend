from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Read the CSV file
data = pd.read_csv('house_prices.csv')

# Columns to convert 1s and 0s to 'yes' and 'no'
convert_to_yes_no = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Convert specified columns
data[convert_to_yes_no] = data[convert_to_yes_no].replace({1: 'yes', 0: 'no'})

# Convert 'furnishingstatus' column
furnishing_mapping = {0: 'unfurnished', 1: 'semi-furnished', 2: 'furnished'}
data['furnishingstatus'] = data['furnishingstatus'].map(furnishing_mapping)

@app.route('/')
def index():
    # Pass the data to the template
    return render_template('index.html', data=data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
