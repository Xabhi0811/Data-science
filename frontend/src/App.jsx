import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [email, setEmail] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const [trainingFile, setTrainingFile] = useState(null);
    const [training, setTraining] = useState(false);
    const [activeTab, setActiveTab] = useState('validate');

    const validateEmail = async () => {
        if (!email) {
            alert('Please enter an email address');
            return;
        }
        
        if (!email.includes('@')) {
            alert('Please enter a valid email address');
            return;
        }
        
        setLoading(true);
        try {
            const response = await axios.post('http://localhost:3001/api/validate-email', { email });
            setResult(response.data);
            fetchHistory();
        } catch (error) {
            console.error('Error validating email:', error);
            alert('Error validating email. Please make sure the server is running.');
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
        if (!trainingFile) {
            alert('Please select a CSV file first');
            return;
        }
        
        setTraining(true);
        const formData = new FormData();
        formData.append('file', trainingFile);
        
        try {
            const response = await axios.post('http://localhost:5000/api/train', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            alert(`🎉 Model trained successfully!\nAccuracy: ${(response.data.accuracy * 100).toFixed(2)}%`);
        } catch (error) {
            console.error('Error training model:', error);
            alert('Error training model. Please check if the Python server is running.');
        }
        setTraining(false);
    };

    const clearResults = () => {
        setResult(null);
        setEmail('');
    };

    useEffect(() => {
        fetchHistory();
    }, []);

    return (
        <div className="App">
            {/* Header */}
            <header className="header">
                <div className="container">
                    <div className="logo">
                        <div className="logo-icon">📧</div>
                        <h1>Email Validator Pro</h1>
                    </div>
                    <p className="tagline">AI-powered email verification system</p>
                </div>
            </header>

            <div className="container">
                {/* Navigation Tabs */}
                <div className="tabs">
                    <button 
                        className={`tab ${activeTab === 'validate' ? 'active' : ''}`}
                        onClick={() => setActiveTab('validate')}
                    >
                        🔍 Validate Email
                    </button>
                    <button 
                        className={`tab ${activeTab === 'train' ? 'active' : ''}`}
                        onClick={() => setActiveTab('train')}
                    >
                        🏋️ Train Model
                    </button>
                    <button 
                        className={`tab ${activeTab === 'history' ? 'active' : ''}`}
                        onClick={() => setActiveTab('history')}
                    >
                        📊 History
                    </button>
                </div>

                {/* Main Content */}
                <div className="main-content">
                    {/* Validation Tab */}
                    {activeTab === 'validate' && (
                        <div className="card">
                            <div className="card-header">
                                <h2>Validate Email Address</h2>
                                <p>Enter an email address to check if it's real or fake</p>
                            </div>
                            
                            <div className="input-container">
                                <div className="input-with-icon">
                                    <span className="input-icon">✉️</span>
                                    <input
                                        type="email"
                                        placeholder="Enter email address (e.g., example@gmail.com)"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        onKeyPress={(e) => e.key === 'Enter' && validateEmail()}
                                        className="email-input"
                                    />
                                </div>
                                <div className="button-group">
                                    <button 
                                        onClick={validateEmail} 
                                        disabled={loading}
                                        className="btn btn-primary"
                                    >
                                        {loading ? (
                                            <>
                                                <span className="spinner"></span>
                                                Analyzing...
                                            </>
                                        ) : (
                                            'Validate Email'
                                        )}
                                    </button>
                                    <button 
                                        onClick={clearResults}
                                        className="btn btn-secondary"
                                    >
                                        Clear
                                    </button>
                                </div>
                            </div>

                            {/* Results */}
                            {result && (
                                <div className={`result-card ${result.is_real ? 'real' : 'fake'}`}>
                                    <div className="result-header">
                                        <div className="result-icon">
                                            {result.is_real ? '✅' : '❌'}
                                        </div>
                                        <div className="result-title">
                                            <h3>{result.is_real ? 'Real Email' : 'Fake Email'}</h3>
                                            <p>{result.email}</p>
                                        </div>
                                        <div className="confidence-badge">
                                            {(result.confidence * 100).toFixed(1)}% confidence
                                        </div>
                                    </div>
                                    
                                    <div className="probability-bars">
                                        <div className="probability-item real">
                                            <label>Real Probability</label>
                                            <div className="progress-bar">
                                                <div 
                                                    className="progress-fill real"
                                                    style={{width: `${result.probability_real * 100}%`}}
                                                ></div>
                                            </div>
                                            <span>{(result.probability_real * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="probability-item fake">
                                            <label>Fake Probability</label>
                                            <div className="progress-bar">
                                                <div 
                                                    className="progress-fill fake"
                                                    style={{width: `${result.probability_fake * 100}%`}}
                                                ></div>
                                            </div>
                                            <span>{(result.probability_fake * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Training Tab */}
                    {activeTab === 'train' && (
                        <div className="card">
                            <div className="card-header">
                                <h2>Train Machine Learning Model</h2>
                                <p>Upload a CSV file with email data to train the AI model</p>
                            </div>
                            
                            <div className="upload-area">
                                <div className="upload-box">
                                    <div className="upload-icon">📁</div>
                                    <h4>Upload CSV Dataset</h4>
                                    <p>File should contain 'email' and 'is_real' columns</p>
                                    <input 
                                        type="file" 
                                        accept=".csv"
                                        onChange={(e) => setTrainingFile(e.target.files[0])}
                                        className="file-input"
                                        id="file-upload"
                                    />
                                    <label htmlFor="file-upload" className="btn btn-outline">
                                        Choose File
                                    </label>
                                    {trainingFile && (
                                        <div className="file-info">
                                            <span>📄 {trainingFile.name}</span>
                                        </div>
                                    )}
                                </div>
                                
                                <button 
                                    onClick={trainModel} 
                                    disabled={!trainingFile || training}
                                    className="btn btn-primary train-btn"
                                >
                                    {training ? (
                                        <>
                                            <span className="spinner"></span>
                                            Training Model...
                                        </>
                                    ) : (
                                        '🚀 Train Model'
                                    )}
                                </button>
                            </div>
                            
                            <div className="training-info">
                                <h4>📋 Required CSV Format:</h4>
                                <div className="code-block">
                                    email,is_real<br/>
                                    john.doe@gmail.com,1<br/>
                                    fake-email@spam.com,0<br/>
                                    alice.smith@company.com,1<br/>
                                    ...
                                </div>
                            </div>
                        </div>
                    )}

                    {/* History Tab */}
                    {activeTab === 'history' && (
                        <div className="card">
                            <div className="card-header">
                                <h2>Validation History</h2>
                                <p>Recent email validation results</p>
                            </div>
                            
                            <div className="history-container">
                                {history.length === 0 ? (
                                    <div className="empty-state">
                                        <div className="empty-icon">📊</div>
                                        <h3>No validation history yet</h3>
                                        <p>Validate some emails to see the history here</p>
                                    </div>
                                ) : (
                                    <div className="history-list">
                                        {history.map((item, index) => (
                                            <div key={index} className={`history-item ${item.isReal ? 'real' : 'fake'}`}>
                                                <div className="history-email">
                                                    <span className="email-text">{item.email}</span>
                                                    <span className="timestamp">
                                                        {new Date(item.timestamp).toLocaleDateString()}
                                                    </span>
                                                </div>
                                                <div className="history-result">
                                                    <span className={`status-badge ${item.isReal ? 'real' : 'fake'}`}>
                                                        {item.isReal ? 'Real' : 'Fake'}
                                                    </span>
                                                    <span className="confidence">
                                                        {(item.confidence * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Footer */}
            <footer className="footer">
                <div className="container">
                    <p>Email Validator Pro • Powered by Machine Learning</p>
                </div>
            </footer>
        </div>
    );
}

export default App;