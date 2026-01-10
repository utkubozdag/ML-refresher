import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_C=1.0.bin'

app = Flask('churn')

with open(model_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def get_probability(customer):
    X = dv.transform([customer])
    customer_prediction = model.predict_proba(X)[0,1]
    churn = customer_prediction >= 0.5
    return customer_prediction, churn

@app.route('/predict', methods= ['POST'])
def predict():
    customer = request.get_json()
    customer_prediction, churn = get_probability(customer)

    result = {
        'churn_probability': float(customer_prediction),
        'churn' : bool(churn)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

    