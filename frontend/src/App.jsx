import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Add axios base URL and error handling
const API_BASE = 'http://localhost:5000';

function App() {
    const [email, setEmail] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const [trainingFile, setTrainingFile] = useState(null);
    const [training, setTraining] = useState(false);
    const [activeTab, setActiveTab] = useState('validate');
    const [serverStatus, setServerStatus] = useState('checking');

    // Check server status on component mount
    useEffect(() => {
        checkServerStatus();
        fetchHistory();
    }, []);

    const checkServerStatus = async () => {
        try {
            const response = await axios.get(`${API_BASE}/api/health`);
            setServerStatus('connected');
            console.log('✅ Server is connected:', response.data);
        } catch (error) {
            setServerStatus('disconnected');
            console.error('❌ Server connection failed:', error);
        }
    };

    const validateEmail = async () => {
        if (!email) {
            alert('Please enter an email address');
            return;
        }
        
        if (!email.includes('@')) {
            alert('Please enter a valid email address');
            return;
        }
        
        if (serverStatus === 'disconnected') {
            alert('Cannot connect to server. Please make sure the Python backend is running on port 5000.');
            return;
        }
        
        setLoading(true);
        try {
            const response = await axios.post(`${API_BASE}/api/validate`, { 
                email: email 
            }, {
                timeout: 10000 // 10 second timeout
            });
            
            if (response.data.error) {
                alert(`Validation error: ${response.data.error}`);
            } else {
                setResult(response.data);
            }
        } catch (error) {
            console.error('Error validating email:', error);
            handleApiError(error, 'validate');
        }
        setLoading(false);
    };

    const trainModel = async () => {
        if (!trainingFile) {
            alert('Please select a CSV file first');
            return;
        }
        
        if (serverStatus === 'disconnected') {
            alert('Cannot connect to server. Please make sure the Python backend is running on port 5000.');
            return;
        }
        
        setTraining(true);
        const formData = new FormData();
        formData.append('file', trainingFile);
        
        try {
            const response = await axios.post(`${API_BASE}/api/train`, formData, {
                headers: { 
                    'Content-Type': 'multipart/form-data' 
                },
                timeout: 30000 // 30 second timeout for training
            });
            
            if (response.data.error) {
                alert(`Training error: ${response.data.error}`);
            } else {
                alert(`🎉 Model trained successfully!\nAccuracy: ${(response.data.accuracy * 100).toFixed(2)}%`);
                // Update server status after successful training
                setServerStatus('connected');
            }
        } catch (error) {
            console.error('Error training model:', error);
            handleApiError(error, 'train');
        }
        setTraining(false);
    };

    const handleApiError = (error, operation) => {
        if (error.code === 'NETWORK_ERROR' || error.code === 'ECONNREFUSED') {
            alert(`Cannot connect to the server. Please make sure:\n\n1. Python backend is running on port 5000\n2. You've installed all required packages\n3. No other service is using port 5000\n\nError: ${error.message}`);
            setServerStatus('disconnected');
        } else if (error.response) {
            // Server responded with error status
            alert(`Server error (${error.response.status}): ${error.response.data.error || error.response.statusText}`);
        } else if (error.request) {
            // Request made but no response received
            alert('No response from server. Please check if the Python backend is running.');
            setServerStatus('disconnected');
        } else {
            // Something else happened
            alert(`Error during ${operation}: ${error.message}`);
        }
    };

    const fetchHistory = async () => {
        try {
            // For now, we'll use localStorage for history since MongoDB might not be set up
            const savedHistory = localStorage.getItem('emailValidationHistory');
            if (savedHistory) {
                setHistory(JSON.parse(savedHistory));
            }
        } catch (error) {
            console.error('Error fetching history:', error);
        }
    };

    const saveToHistory = (result) => {
        const newHistoryItem = {
            email: result.email,
            isReal: result.is_real,
            confidence: result.confidence,
            timestamp: new Date().toISOString()
        };
        
        const updatedHistory = [newHistoryItem, ...history.slice(0, 49)]; // Keep last 50 items
        setHistory(updatedHistory);
        localStorage.setItem('emailValidationHistory', JSON.stringify(updatedHistory));
    };

    const clearResults = () => {
        setResult(null);
        setEmail('');
    };

    // Update history when new result comes in
    useEffect(() => {
        if (result && !result.error) {
            saveToHistory(result);
        }
    }, [result]);

    return (
        <div className="App">
            {/* Server Status Indicator */}
            <div className={`server-status ${serverStatus}`}>
                Server: {serverStatus === 'connected' ? '✅ Connected' : '❌ Disconnected'}
                <button onClick={checkServerStatus} className="refresh-btn">
                    🔄
                </button>
            </div>

            {/* Rest of your JSX remains the same */}
            <header className="header">
                <div className="container">
                    <div className="logo">
                        <div className="logo-icon">📧</div>
                        <h1>Email Validator Pro</h1>
                    </div>
                    <p className="tagline">AI-powered email verification system</p>
                </div>
            </header>

            {/* ... rest of your existing JSX ... */}
        </div>
    );
}

export default App;