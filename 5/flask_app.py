from flask import Flask, request, jsonify
import pickle

with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    client = request.get_json()

    X_test = dv.transform([client])
    credit_acceptance_probability = model.predict_proba(X_test)[0][1]
    credit_acceptance = credit_acceptance_probability >= 0.5

    result = {
        "credit_acceptance_probability": round(credit_acceptance_probability, 3),
        "credit_acceptance": bool(credit_acceptance),  # Boolean conversion is must to jsonify
    }

    return jsonify(result)
