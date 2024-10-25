from flask import Flask, request, jsonify
import pickle
import logging


def load_pickle(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except (EOFError, FileNotFoundError, pickle.UnpicklingError) as e:
        logging.error(f"Error loading {filename}: {e}")
        return None


model = load_pickle('model1.bin')
dv = load_pickle('dv.bin')

app = Flask('app')


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not dv:
        return jsonify({'error': 'Model or DV not loaded properly'}), 500

    client = request.get_json()
    if not client:
        return jsonify({'error': 'Invalid input data'}), 400

    try:
        client_features = dv.transform([client])
        y_pred = model.predict_proba(client_features)[0, 1]
        return jsonify({'Subscription Probability': y_pred})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
