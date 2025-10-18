from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)
CORS(app)  # This allows all origins

class EmailValidator:
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def extract_features(self, email):
        """Extract features from email address"""
        features = []
        
        # Basic email features
        features.append(len(email))
        features.append(1 if '.' in email else 0)
        features.append(1 if '-' in email else 0)
        features.append(1 if '_' in email else 0)
        features.append(1 if any(char.isdigit() for char in email) else 0)
        
        # Domain features
        if '@' in email:
            domain = email.split('@')[-1]
            features.append(len(domain))
            features.append(1 if any(d in domain for d in ['gmail', 'yahoo', 'hotmail', 'outlook', 'edu']) else 0)
        else:
            features.extend([0, 0])
        
        return features
    
    def train(self, csv_file):
        """Train the model on dataset"""
        try:
            # Load dataset
            df = pd.read_csv(csv_file)
            
            # Prepare features and labels
            features = []
            labels = []
            
            for _, row in df.iterrows():
                email = row['email']
                is_real = row['is_real']
                
                feature_vector = self.extract_features(email)
                features.append(feature_vector)
                labels.append(is_real)
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            joblib.dump(self.model, 'email_validator_model.pkl')
            self.is_trained = True
            
            print(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            raise e
    
    def predict(self, email):
        """Predict if email is real or fake"""
        try:
            if not self.is_trained and os.path.exists('email_validator_model.pkl'):
                self.model = joblib.load('email_validator_model.pkl')
                self.is_trained = True
            
            if not self.is_trained:
                return {"error": "Model not trained. Please train the model first."}
            
            features = self.extract_features(email)
            prediction = self.model.predict([features])[0]
            probability = self.model.predict_proba([features])[0]
            
            return {
                'email': email,
                'is_real': bool(prediction),
                'confidence': float(max(probability)),
                'probability_real': float(probability[1]),
                'probability_fake': float(probability[0])
            }
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# Initialize validator
validator = EmailValidator()

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        print("📥 Received training request...")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Save uploaded file
        file_path = 'uploaded_dataset.csv'
        file.save(file_path)
        print(f"📁 File saved: {file_path}")
        
        # Check if file has required columns
        try:
            df = pd.read_csv(file_path)
            if 'email' not in df.columns or 'is_real' not in df.columns:
                return jsonify({'error': 'CSV must contain "email" and "is_real" columns'}), 400
            print(f"📊 Dataset loaded: {len(df)} rows")
        except Exception as e:
            return jsonify({'error': f'Invalid CSV file: {str(e)}'}), 400
        
        # Train model
        accuracy = validator.train(file_path)
        
        return jsonify({
            'message': 'Model trained successfully',
            'accuracy': accuracy,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def validate_email():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        email = data.get('email')
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        print(f"🔍 Validating email: {email}")
        result = validator.predict(email)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_trained': validator.is_trained,
        'message': 'Python Flask server is running!'
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'Email Validator API is running!',
        'endpoints': {
            'GET /api/health': 'Check server status',
            'POST /api/train': 'Train model with CSV file',
            'POST /api/validate': 'Validate single email'
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Email Validator Server...")
    print("📧 API Endpoints:")
    print("   GET  /api/health - Health check")
    print("   POST /api/train - Train model")
    print("   POST /api/validate - Validate email")
    print("   GET  / - Server info")
    
    app.run(debug=True, host='0.0.0.0', port=5000)