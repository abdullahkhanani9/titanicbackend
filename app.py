from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Read the CSV file
data = pd.read_csv('house_prices.csv')

@app.route('/')
def index():
    # Pass the data to the template
    return render_template('index.html', data=data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
