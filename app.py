from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Form values
        input_data = {
            'longitude': float(request.form['longitude']),
            'latitude': float(request.form['latitude']),
            'housing_median_age': float(request.form['housing_median_age']),
            'total_rooms': float(request.form['total_rooms']),
            'total_bedrooms': float(request.form['total_bedrooms']),
            'population': float(request.form['population']),
            'households': float(request.form['households']),
            'median_income': float(request.form['median_income']),
            'ocean_proximity': request.form['ocean_proximity']
        }

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        return render_template('index.html', prediction_text=f'Predicted House Price: ${round(prediction, 2)}')

if __name__ == '__main__':
    app.run(debug=True)
