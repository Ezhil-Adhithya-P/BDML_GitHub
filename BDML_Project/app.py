from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved logistic regression model
with open("model/creditcard_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define feature order (as per your dataset)
FEATURE_ORDER = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                 'V11','V12','V13','V14','V15','V16','V17','V18',
                 'V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_ORDER)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        features = np.array(data).reshape(1, -1)
        prediction = model.predict(features)[0]

        result = "Fraudulent Transaction ⚠️" if prediction == 1 else "Legitimate Transaction ✅"
        return render_template('index.html', prediction_text=result, features=FEATURE_ORDER)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", features=FEATURE_ORDER)

if __name__ == '__main__':
    app.run(debug=True)
