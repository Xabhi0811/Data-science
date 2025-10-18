import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import joblib
import os

class EmailValidator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=1000)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def extract_features(self, email):
        """Extract features from email address"""
        features = {}
        
        # Basic email features
        features['length'] = len(email)
        features['has_dot'] = 1 if '.' in email else 0
        features['has_hyphen'] = 1 if '-' in email else 0
        features['has_underscore'] = 1 if '_' in email else 0
        features['has_numbers'] = 1 if any(char.isdigit() for char in email) else 0
        
        # Domain features
        domain = email.split('@')[-1] if '@' in email else ''
        features['domain_length'] = len(domain)
        features['has_popular_domain'] = 1 if any(d in domain for d in ['gmail', 'yahoo', 'hotmail', 'outlook']) else 0
        
        # Pattern features
        features['special_char_count'] = len(re.findall(r'[^a-zA-Z0-9@._-]', email))
        features['consecutive_special'] = 1 if re.search(r'[._-]{2,}', email) else 0
        
        return features
    
    def prepare_data(self, emails):
        """Prepare data for training/prediction"""
        features_list = []
        for email in emails:
            features = self.extract_features(email)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def train(self, csv_file):
        """Train the model on dataset"""
        # Load dataset
        df = pd.read_csv(csv_file)
        
        # Prepare features
        X = self.prepare_data(df['email'])
        y = df['is_real']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        
        # Save model
        joblib.dump(self.model, 'email_validator_model.pkl')
        return accuracy
    
    def predict(self, email):
        """Predict if email is real or fake"""
        if not self.is_trained:
            # Load saved model if exists
            if os.path.exists('email_validator_model.pkl'):
                self.model = joblib.load('email_validator_model.pkl')
                self.is_trained = True
            else:
                return {"error": "Model not trained"}
        
        features = self.prepare_data([email])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'email': email,
            'is_real': bool(prediction),
            'confidence': float(max(probability)),
            'probability_real': float(probability[1]),
            'probability_fake': float(probability[0])
        }

# Initialize validator
validator = EmailValidator()