from flask import Flask, request, jsonify
from flask_cors import CORS
from model import validator
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            file.save('uploaded_dataset.csv')
            accuracy = validator.train('uploaded_dataset.csv')
            return jsonify({
                'message': 'Model trained successfully',
                'accuracy': accuracy
            })
        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def validate_email():
    try:
        data = request.json
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        result = validator.predict(email)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-validate', methods=['POST'])
def batch_validate():
    try:
        data = request.json
        emails = data.get('emails', [])
        
        if not emails:
            return jsonify({'error': 'Emails are required'}), 400
        
        results = []
        for email in emails:
            result = validator.predict(email)
            results.append(result)
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_trained': validator.is_trained})

if __name__ == '__main__':
    app.run(debug=True, port=5000)