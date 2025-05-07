
# Housing Price Prediction Web App

This project is a machine learning web application that predicts housing prices based on several features, such as geographical data, median income, number of rooms, etc. The model is built using a dataset of California housing prices and deployed via a Flask web app.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How to Run Locally](#how-to-run-locally)
- [Usage](#usage)
- [Deployment](#deployment)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This web app uses a **XGBoost model** trained on the California Housing Prices dataset to predict median housing prices. The user can input various features (e.g., location, number of rooms, etc.), and the model will predict the house price.

### Features:
- **Data Preprocessing**: The data is cleaned and scaled, and categorical variables are encoded.
- **Model Training**: We use multiple models (Linear Regression, Random Forest, XGBoost) to predict the house price.
- **Web Interface**: A simple web app built with Flask allows users to interact with the model and get predictions.
- **Deployment**: The app can be deployed on cloud platforms like Heroku, AWS, GCP, etc.

---

## Technologies Used

- **Python 3.x**
- **Flask**: Web framework for building the application.
- **XGBoost**: The main machine learning model used to predict house prices.
- **scikit-learn**: Used for data preprocessing, model evaluation, and training.
- **pandas & numpy**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization.
- **joblib**: For saving and loading the trained model.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/housing-price-prediction.git
cd housing-price-prediction
```

### 2. Create a virtual environment

If you're using `venv`, you can create a virtual environment with the following command:

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For Windows:
```bash
.env\Scriptsctivate
```

For macOS/Linux:
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run Locally

### 1. Train the Model

Before running the web app, you need to train the model and save it. You can do this by running `train_model.py`:

```bash
python train_model.py
```

This will train the model, save it as `model.pkl`, and also save the scaler used to preprocess the input features.

### 2. Run the Flask App

Now that the model is trained, you can start the Flask app:

```bash
python app.py
```

The app will run locally on `http://127.0.0.1:5000/`. Open this URL in your browser.

---

## Usage

1. **Input Features**: On the homepage, fill in the required details, such as `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, and `ocean_proximity`.
2. **Submit the Form**: After filling in the form, click on the "Predict Price" button to submit the input.
3. **Prediction Result**: The predicted house price will be displayed on the screen.

---

## Deployment

This web app can be deployed on various platforms. Here is how to deploy on **Heroku**:

### 1. Install Heroku CLI

Follow the instructions from the [Heroku CLI documentation](https://devcenter.heroku.com/articles/heroku-cli) to install it on your system.

### 2. Create a `Procfile` and `requirements.txt`

Ensure you have a `Procfile` and a `requirements.txt` for Heroku to recognize your app.

- **Procfile**:
    ```
    web: python app.py
    ```

- **requirements.txt**:
    Run the following to generate a `requirements.txt`:
    ```bash
    pip freeze > requirements.txt
    ```

### 3. Push to Heroku

1. Log in to Heroku:
    ```bash
    heroku login
    ```
   
2. Create a new Heroku app:
    ```bash
    heroku create your-app-name
    ```

3. Push your code to Heroku:
    ```bash
    git push heroku master
    ```

4. Open the app:
    ```bash
    heroku open
    ```

---

## Folder Structure

Here is the recommended folder structure for this project:

```
housing-price-prediction/
│
├── app.py                    # Flask web app
├── train_model.py            # Model training script
├── model.pkl                 # Trained model file
├── scaler.pkl                # Scaler for feature scaling
├── requirements.txt          # Python dependencies
├── Procfile                  # Heroku process file
├── runtime.txt               # (Optional) Python version specification
├── templates/                # HTML templates (Flask will look here for index.html)
│   └── index.html            # Web interface for input
└── static/                   # Static files (CSS, JS, images)
    └── styles.css            # Optional custom styles
```

---

## Contributing

1. **Fork** the repository.
2. **Clone** the forked repository to your local machine.
3. **Create a new branch** (`git checkout -b feature-branch`).
4. **Make your changes** and commit them (`git commit -am 'Added new feature'`).
5. **Push** to your fork (`git push origin feature-branch`).
6. **Create a pull request** on GitHub.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

