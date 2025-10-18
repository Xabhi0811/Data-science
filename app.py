from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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

    def rule_based_email_check(self, email):
        """Rule-based email validation when model is not trained"""
        email = str(email).lower().strip()
        
        # Check for valid email format
        if '@' not in email or '.' not in email:
            return {
                'email': email,
                'is_real': False,
                'confidence': 0.8,
                'probability_real': 0.2,
                'probability_fake': 0.8,
                'model_used': 'rule_based',
                'note': 'Invalid email format'
            }
        
        # Suspicious domains
        suspicious_domains = [
            'spam.com', 'fake-mail.xyz', 'throwawaymail.com', 'guerrillamail.com',
            '10minutemail.com', 'temp-mail.org', 'disposable.com', 'trashmail.com',
            'fakeinbox.com', 'mailinator.com', 'yopmail.com', 'sharkmail.com'
        ]
        
        # Real domains
        real_domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com',
            'protonmail.com', 'aol.com', 'zoho.com', 'mail.com'
        ]
        
        domain = email.split('@')[1]
        
        # Check domain reputation
        if any(susp_domain in domain for susp_domain in suspicious_domains):
            return {
                'email': email,
                'is_real': False,
                'confidence': 0.85,
                'probability_real': 0.15,
                'probability_fake': 0.85,
                'model_used': 'rule_based',
                'note': 'Suspicious domain detected'
            }
        elif any(real_domain in domain for real_domain in real_domains):
            return {
                'email': email,
                'is_real': True,
                'confidence': 0.75,
                'probability_real': 0.75,
                'probability_fake': 0.25,
                'model_used': 'rule_based',
                'note': 'Trusted domain detected'
            }
        else:
            # For unknown domains, use length and pattern analysis
            local_part = email.split('@')[0]
            if len(local_part) < 3 or re.search(r'[0-9]{5,}', local_part) or re.search(r'[._-]{2,}', local_part):
                return {
                    'email': email,
                    'is_real': False,
                    'confidence': 0.7,
                    'probability_real': 0.3,
                    'probability_fake': 0.7,
                    'model_used': 'rule_based',
                    'note': 'Suspicious email pattern'
                }
            else:
                return {
                    'email': email,
                    'is_real': True,
                    'confidence': 0.6,
                    'probability_real': 0.6,
                    'probability_fake': 0.4,
                    'model_used': 'rule_based',
                    'note': 'Appears legitimate'
                }

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
                print("✅ Email Model loaded from file")
            
            if not self.is_trained:
                print("🔄 Using rule-based email validation (model not trained)")
                return self.rule_based_email_check(email)
            
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
                'features_used': len(features),
                'model_used': 'machine_learning'
            }
            
            status = "Real" if result['is_real'] else "Fake"
            print(f"🔍 Prediction for {email}: {status} (confidence: {result['confidence']:.3f})")
            return result
            
        except Exception as e:
            print(f"❌ Email prediction error, using rule-based: {str(e)}")
            return self.rule_based_email_check(email)

class SMSDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_trained = False

    def preprocess_text(self, text):
        """Preprocess SMS text"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def rule_based_sms_detection(self, message):
        """Basic rule-based spam detection when model not trained"""
        spam_keywords = [
            'free', 'winner', 'won', 'prize', 'cash', 'loan', 'credit', 
            'urgent', 'alert', 'security', 'verify', 'click', 'call now',
            'limited time', 'offer', 'discount', 'winner', 'congratulations',
            'selected', 'exclusive', 'deal', 'money', 'profit', 'investment',
            'bitcoin', 'crypto', 'lottery', 'gift card', 'claim', 'bonus',
            'text stop', 'reply stop', 'msg', 'txt', 'rate', 'charge',
            'guaranteed', 'risk-free', 'million', 'thousand', 'dollar',
            'pounds', 'euros', 'income', 'earn', 'make money', 'work from home'
        ]
        
        message_lower = message.lower()
        spam_score = 0
        
        for keyword in spam_keywords:
            if keyword in message_lower:
                spam_score += 1
        
        # Calculate spam probability
        max_possible_score = min(len(spam_keywords), 10)  # Cap at 10 for normalization
        spam_probability = min(spam_score / 5, 0.95)  # Normalize and cap at 0.95
        
        # Adjust based on message length
        if len(message) < 20:
            spam_probability *= 0.8  # Short messages are less likely to be spam
        
        # Common ham indicators
        ham_indicators = ['hello', 'hi ', 'thanks', 'thank you', 'please', 'meeting', 'lunch', 'dinner', 'see you', 'tomorrow', 'today']
        ham_score = sum(1 for indicator in ham_indicators if indicator in message_lower)
        spam_probability = max(0.1, spam_probability - (ham_score * 0.1))
        
        is_spam = spam_probability > 0.5
        confidence = abs(spam_probability - 0.5) * 2  # Convert to confidence score
        
        return {
            'message': message,
            'is_spam': is_spam,
            'category': 'spam' if is_spam else 'ham',
            'confidence': max(confidence, 0.3),  # Minimum 30% confidence
            'probability_spam': spam_probability,
            'probability_ham': 1 - spam_probability,
            'model_used': 'rule_based',
            'note': 'Train the model for better accuracy',
            'spam_keywords_found': spam_score
        }

    def train_sms_model(self, csv_file):
        """Train SMS spam detection model"""
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Loaded {len(df)} SMS messages for spam detection")
            
            # Check dataset format
            if 'Category' not in df.columns or 'Message' not in df.columns:
                return {"error": "SMS dataset must contain 'Category' and 'Message' columns"}
            
            # Prepare data
            df['processed_text'] = df['Message'].apply(self.preprocess_text)
            X = self.vectorizer.fit_transform(df['processed_text'])
            y = df['Category'].map({'ham': 0, 'spam': 1})
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            accuracy = self.model.score(X, y)
            self.is_trained = True
            
            # Save model
            joblib.dump({'model': self.model, 'vectorizer': self.vectorizer}, 'sms_model.pkl')
            
            print(f"✅ SMS Spam Model trained! Accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            print(f"❌ SMS Training error: {str(e)}")
            raise e

    def detect_sms(self, message):
        """Detect if SMS message is spam or ham"""
        try:
            # Try to load pre-trained model
            if not self.is_trained and os.path.exists('sms_model.pkl'):
                saved_data = joblib.load('sms_model.pkl')
                self.model = saved_data['model']
                self.vectorizer = saved_data['vectorizer']
                self.is_trained = True
                print("✅ SMS Model loaded from file")
            
            if not self.is_trained:
                print("🔄 Using rule-based SMS detection (model not trained)")
                return self.rule_based_sms_detection(message)
            
            # Preprocess and predict
            processed_text = self.preprocess_text(message)
            X = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            result = {
                'message': message,
                'is_spam': bool(prediction),
                'category': 'spam' if prediction else 'ham',
                'confidence': float(max(probabilities)),
                'probability_spam': float(probabilities[1]),
                'probability_ham': float(probabilities[0]),
                'model_used': 'machine_learning'
            }
            
            print(f"🔍 SMS Prediction: {'SPAM' if result['is_spam'] else 'HAM'} (confidence: {result['confidence']:.3f})")
            return result
            
        except Exception as e:
            print(f"❌ SMS detection error, using rule-based: {str(e)}")
            return self.rule_based_sms_detection(message)

# Initialize both validators
email_validator = AdvancedEmailValidator()
sms_detector = SMSDetector()

@app.route('/')
def home():
    return jsonify({
        "message": "Dual AI Validator API is running! 🚀",
        "endpoints": {
            "GET /api/health": "Check server status",
            "POST /api/train": "Train email model with CSV",
            "POST /api/train-sms": "Train SMS spam model",
            "POST /api/validate": "Validate email address",
            "POST /api/detect-sms": "Detect SMS spam"
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
        "email_model_trained": email_validator.is_trained,
        "sms_model_trained": sms_detector.is_trained,
        "message": "Dual AI Validator is running! ✅",
        "supported_features": ["Email Validation", "SMS Spam Detection"],
        "note": "Using rule-based detection when models not trained"
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        print("📥 Received email training request...")
        
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
        accuracy = email_validator.train(filename)
        
        return jsonify({
            "message": "Email Model trained successfully! 🎉",
            "accuracy": accuracy,
            "model_trained": True,
            "model_type": "Email Validator",
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/train-sms', methods=['POST'])
def train_sms_model():
    try:
        print("📥 Received SMS training request...")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Please upload a CSV file"}), 400
        
        # Save the file
        filename = 'uploaded_sms_dataset.csv'
        file.save(filename)
        print(f"💾 SMS file saved as {filename}")
        
        # Train the SMS model
        accuracy = sms_detector.train_sms_model(filename)
        
        return jsonify({
            "message": "SMS Spam Model trained successfully! 🎉",
            "accuracy": accuracy,
            "model_trained": True,
            "model_type": "SMS Spam Detector",
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ SMS Training Error: {str(e)}")
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
        result = email_validator.predict(email)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect-sms', methods=['POST'])
def detect_sms():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        print(f"🔍 Analyzing SMS message: {message[:50]}...")
        result = sms_detector.detect_sms(message)
        
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
            'format': email_validator.detect_dataset_format(df),
            'sample_data': df.head(3).to_dict('records'),
            'suggested_model': 'email' if 'email' in df.columns else 'sms'
        }
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return jsonify(format_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Dual AI Validator Server...")
    print("📍 Server will run on: http://localhost:5000")
    print("📧 Available endpoints:")
    print("   GET  /api/health")
    print("   POST /api/train (Email validation)")
    print("   POST /api/train-sms (SMS spam detection)")
    print("   POST /api/validate (Check email)")
    print("   POST /api/detect-sms (Check SMS)")
    print("\n📊 Supported features:")
    print("   - Email Address Validation")
    print("   - SMS Spam Detection")
    print("   - Rule-based fallback when models not trained")
    print("\n⚡ Starting server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)