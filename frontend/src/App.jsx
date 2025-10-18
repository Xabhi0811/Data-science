import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [email, setEmail] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const [trainingFile, setTrainingFile] = useState(null);

    const validateEmail = async () => {
        if (!email) return;
        
        setLoading(true);
        try {
            const response = await axios.post('http://localhost:3001/api/validate-email', { email });
            setResult(response.data);
            fetchHistory();
        } catch (error) {
            console.error('Error validating email:', error);
            alert('Error validating email');
        }
        setLoading(false);
    };

    const fetchHistory = async () => {
        try {
            const response = await axios.get('http://localhost:3001/api/validation-history');
            setHistory(response.data);
        } catch (error) {
            console.error('Error fetching history:', error);
        }
    };

    const trainModel = async () => {
        if (!trainingFile) return;
        
        const formData = new FormData();
        formData.append('file', trainingFile);
        
        try {
            const response = await axios.post('http://localhost:5000/api/train', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            alert(`Model trained successfully! Accuracy: ${response.data.accuracy}`);
        } catch (error) {
            console.error('Error training model:', error);
            alert('Error training model');
        }
    };

    useEffect(() => {
        fetchHistory();
    }, []);

    return (
        <div className="App">
            <div className="container">
                <h1>Email Validator</h1>
                <p>Check if an email is real or fake using Machine Learning</p>
                
                {/* Training Section */}
                <div className="section">
                    <h2>Train Model</h2>
                    <input 
                        type="file" 
                        accept=".csv"
                        onChange={(e) => setTrainingFile(e.target.files[0])}
                    />
                    <button onClick={trainModel}>Train Model</button>
                </div>

                {/* Validation Section */}
                <div className="section">
                    <h2>Validate Email</h2>
                    <div className="input-group">
                        <input
                            type="email"
                            placeholder="Enter email to validate"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && validateEmail()}
                        />
                        <button onClick={validateEmail} disabled={loading}>
                            {loading ? 'Validating...' : 'Validate'}
                        </button>
                    </div>
                </div>

                {/* Results */}
                {result && (
                    <div className={`result ${result.is_real ? 'real' : 'fake'}`}>
                        <h3>Validation Result</h3>
                        <p><strong>Email:</strong> {result.email}</p>
                        <p><strong>Status:</strong> {result.is_real ? '✅ Real Email' : '❌ Fake Email'}</p>
                        <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
                        <p><strong>Real Probability:</strong> {(result.probability_real * 100).toFixed(2)}%</p>
                        <p><strong>Fake Probability:</strong> {(result.probability_fake * 100).toFixed(2)}%</p>
                    </div>
                )}

                {/* History */}
                <div className="section">
                    <h2>Validation History</h2>
                    <div className="history">
                        {history.map((item, index) => (
                            <div key={index} className={`history-item ${item.isReal ? 'real' : 'fake'}`}>
                                <span>{item.email}</span>
                                <span>{item.isReal ? 'Real' : 'Fake'}</span>
                                <span>{(item.confidence * 100).toFixed(1)}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;