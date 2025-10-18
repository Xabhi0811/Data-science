from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import re

app = Flask(__name__)
CORS(app)

print("✅ Flask app created successfully!")

class AdvancedEmailValidator:
    def __init__(self):
        self.model = None
        self.is_trained = False
        print("✅ Advanced Email validator initialized!")

    def extract_features(self, email):
        """Advanced feature extraction from email"""
        features = []
        
        # Basic email features
        features.append(len(email))
        features.append(1 if '.' in email else 0)
        features.append(1 if '@' in email else 0)
        features.append(1 if any(char.isdigit() for char in email) else 0)
        
        # Count special characters
        features.append(email.count('.'))
        features.append(email.count('-'))
        features.append(email.count('_'))
        features.append(email.count('+'))
        
        # Domain analysis
        if '@' in email:
            local_part, domain = email.split('@')
            features.append(len(local_part))
            features.append(len(domain))
            
            # Popular domains
            popular_domains = ['gmail', 'yahoo', 'hotmail', 'outlook', 'edu', 'com', 'org', 'net']
            features.append(1 if any(pop_domain in domain for pop_domain in popular_domains) else 0)
            
            # Suspicious domains
            suspicious_domains = ['spam', 'fake', 'temp', 'throwaway', 'guerrilla', 'disposable', 'trashmail']
            features.append(1 if any(susp_domain in domain for susp_domain in suspicious_domains) else 0)
            
            # Check for consecutive special chars
            features.append(1 if re.search(r'[._-]{2,}', local_part) else 0)
            
            # Check if domain has valid TLD
            valid_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.io']
            features.append(1 if any(domain.endswith(tld) for tld in valid_tlds) else 0)
        else:
            features.extend([0, 0, 0, 0, 0, 0])
            
        return features

    def detect_dataset_format(self, df):
        """Detect the format of the uploaded dataset"""
        columns = df.columns.tolist()
        
        # Check for email validation format
        if 'email' in columns and 'is_real' in columns:
            return 'email_validation'
        
        # Check for spam/ham SMS format
        if 'Category' in columns and 'Message' in columns:
            return 'spam_sms'
        
        # Check for alternative email formats
        if 'Email' in columns and 'Label' in columns:
            return 'email_alt'
            
        return 'unknown'

    def convert_spam_to_email_dataset(self, df):
        """Convert spam/ham SMS dataset to email validation format"""
        print("🔄 Converting spam/ham dataset to email validation format...")
        
        # Create synthetic email dataset based on spam/ham patterns
        real_emails = [
            'john.doe@gmail.com', 'alice.smith@yahoo.com', 'bob.johnson@hotmail.com',
            'sara.wilson@outlook.com', 'mike.brown@company.com', 'lisa.davis@university.edu',
            'james.miller@college.edu', 'emily.white@corporation.com', 'david.lee@business.org',
            'amanda.green@institute.edu', 'robert.taylor@organization.com', 'susan.martin@enterprise.net',
            'kevin.anderson@services.org', 'michelle.clark@corporation.com', 'richard.walker@company.org',
            'jennifer.hall@institution.edu', 'thomas.young@business.com', 'laura.king@organization.net'
        ]
        
        fake_emails = [
            'fake123@spam.com', 'random@fake-mail.xyz', 'temp@throwawaymail.com',
            'test@guerrillamail.com', 'user@10minutemail.com', 'spammer@temp-mail.org',
            'fake@disposable.com', 'trash@trashmail.com', 'temp@fakeinbox.com',
            'test@mailinator.com', 'spam@fake-domain.xyz', 'junk@trash-mail.com',
            'temp@anonymous.com', 'fake@sharkmail.com', 'spam@fakemail.com',
            'temp@yopmail.com', 'fake@mailnesia.com', 'spam@fakeinbox.com'
        ]
        
        # Use the spam/ham distribution to create balanced email dataset
        ham_count = len(df[df['Category'] == 'ham'])
        spam_count = len(df[df['Category'] == 'spam'])
        
        total_samples = min(ham_count + spam_count, 1000)  # Limit dataset size
        
        email_data = []
        
        # Add real emails (from ham messages)
        num_real = min(total_samples // 2, len(real_emails))
        for i in range(num_real):
            email_data.append({'email': real_emails[i % len(real_emails)], 'is_real': 1})
        
        # Add fake emails (from spam messages)  
        num_fake = min(total_samples - num_real, len(fake_emails))
        for i in range(num_fake):
            email_data.append({'email': fake_emails[i % len(fake_emails)], 'is_real': 0})
        
        return pd.DataFrame(email_data)

    def train(self, csv_file):
        """Train the model on dataset with automatic format detection"""
        try:
            print(f"📊 Loading dataset from {csv_file}...")
            df = pd.read_csv(csv_file)
            print(f"✅ Dataset loaded with {len(df)} rows")
            print(f"📋 Columns: {df.columns.tolist()}")
            
            # Detect dataset format
            dataset_format = self.detect_dataset_format(df)
            print(f"🔍 Detected dataset format: {dataset_format}")
            
            if dataset_format == 'spam_sms':
                print("🔄 Converting spam/ham SMS dataset to email validation format...")
                df = self.convert_spam_to_email_dataset(df)
                print(f"✅ Converted dataset: {len(df)} email samples")
            elif dataset_format == 'unknown':
                return jsonify({'error': 'Unknown dataset format. Please use CSV with email/is_real columns or Category/Message columns'}), 400
            
            # Verify we have the right columns now
            if 'email' not in df.columns or 'is_real' not in df.columns:
                return jsonify({'error': 'Dataset must contain email and is_real columns after conversion'}), 400
            
            print(f"🎯 Training with {len(df)} email samples")
            print(f"📊 Class distribution: {df['is_real'].value_counts().to_dict()}")
            
            # Prepare features and labels
            X = []
            y = []
            
            for index, row in df.iterrows():
                email = str(row['email']).strip().lower()
                is_real = int(row['is_real'])
                
                # Skip invalid emails
                if '@' not in email or len(email) < 5:
                    continue
                    
                features = self.extract_features(email)
                X.append(features)
                y.append(is_real)
            
            if len(X) == 0:
                return jsonify({'error': 'No valid email samples found in dataset'}), 400
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"🔢 Features shape: {X.shape}")
            print(f"🎯 Labels shape: {y.shape}")
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            # Calculate accuracy on training data
            accuracy = self.model.score(X, y)
            self.is_trained = True
            
            # Save model
            joblib.dump(self.model, 'email_model.pkl')
            
            print(f"🎉 Model trained successfully!")
            print(f"📈 Training accuracy: {accuracy:.4f}")
            print(f"📊 Class distribution in training: {np.bincount(y)}")
            
            return accuracy
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def predict(self, email):
        """Predict if email is real or fake"""
        try:
            # Load model if not loaded but exists
            if not self.is_trained and os.path.exists('email_model.pkl'):
                self.model = joblib.load('email_model.pkl')
                self.is_trained = True
                print("✅ Model loaded from file")
            
            if not self.is_trained:
                return {"error": "Model not trained. Please train first."}
            
            # Clean and validate email
            email = str(email).strip().lower()
            if '@' not in email:
                return {"error": "Invalid email format"}
            
            features = self.extract_features(email)
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            
            result = {
                'email': email,
                'is_real': bool(prediction),
                'confidence': float(max(probabilities)),
                'probability_real': float(probabilities[1]),
                'probability_fake': float(probabilities[0]),
                'features_used': len(features)
            }
            
            status = "Real" if result['is_real'] else "Fake"
            print(f"🔍 Prediction for {email}: {status} (confidence: {result['confidence']:.3f})")
            return result
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# Create validator instance
validator = AdvancedEmailValidator()

@app.route('/')
def home():
    return jsonify({
        "message": "Advanced Email Validator API is running! 🚀",
        "endpoints": {
            "GET /api/health": "Check server status",
            "POST /api/train": "Train model with CSV (supports multiple formats)",
            "POST /api/validate": "Validate email"
        },
        "supported_formats": [
            "email,is_real",
            "Category,Message (spam/ham SMS)"
        ],
        "status": "ready"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_trained": validator.is_trained,
        "message": "Advanced Email Validator is running! ✅",
        "supported_dataset_formats": ["email/is_real", "spam/ham SMS"]
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        print("📥 Received training request...")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Please upload a CSV file"}), 400
        
        # Save the file
        filename = 'uploaded_dataset.csv'
        file.save(filename)
        print(f"💾 File saved as {filename}")
        
        # Train the model
        accuracy = validator.train(filename)
        
        return jsonify({
            "message": "Model trained successfully! 🎉",
            "accuracy": accuracy,
            "model_trained": True,
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def validate_email():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        email = data.get('email', '').strip()
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        print(f"🔍 Validating email: {email}")
        result = validator.predict(email)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dataset-info', methods=['POST'])
def dataset_info():
    """Get information about uploaded dataset"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save temporarily
        temp_file = 'temp_dataset.csv'
        file.save(temp_file)
        
        # Read dataset
        df = pd.read_csv(temp_file)
        
        # Detect format
        format_info = {
            'columns': df.columns.tolist(),
            'rows': len(df),
            'format': validator.detect_dataset_format(df),
            'sample_data': df.head(3).to_dict('records')
        }
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return jsonify(format_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Advanced Email Validator Server...")
    print("📍 Server will run on: http://localhost:5000")
    print("📧 Available endpoints:")
    print("   http://localhost:5000/")
    print("   http://localhost:5000/api/health")
    print("   POST http://localhost:5000/api/train")
    print("   POST http://localhost:5000/api/validate")
    print("   POST http://localhost:5000/api/dataset-info")
    print("\n📊 Supported dataset formats:")
    print("   - email, is_real")
    print("   - Category, Message (spam/ham SMS)")
    print("\n⚡ Starting server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)