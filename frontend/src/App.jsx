import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:5000';

function App() {
    const [email, setEmail] = useState('');
    const [message, setMessage] = useState('');
    const [result, setResult] = useState(null);
    const [smsResult, setSmsResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [smsLoading, setSmsLoading] = useState(false);
    const [history, setHistory] = useState([]);
    const [trainingFile, setTrainingFile] = useState(null);
    const [training, setTraining] = useState(false);
    const [activeTab, setActiveTab] = useState('validate');
    const [serverStatus, setServerStatus] = useState('checking');
    const [dragOver, setDragOver] = useState(false);

    useEffect(() => {
        checkServerStatus();
        fetchHistory();
    }, []);

    const checkServerStatus = async () => {
        try {
            const response = await axios.get(`${API_BASE}/api/health`);
            setServerStatus('connected');
        } catch (error) {
            setServerStatus('disconnected');
        }
    };

    // Email Validation Functions (Keep existing)
    const validateEmail = async () => {
        if (!email) {
            showToast('Please enter an email address', 'warning');
            return;
        }
        
        if (!email.includes('@')) {
            showToast('Please enter a valid email address', 'warning');
            return;
        }
        
        setLoading(true);
        try {
            const response = await axios.post(`${API_BASE}/api/validate`, { email });
            
            if (response.data.error) {
                showToast(`Validation error: ${response.data.error}`, 'error');
            } else {
                setResult(response.data);
                saveToHistory(response.data);
                showToast('Email validated successfully!', 'success');
            }
        } catch (error) {
            handleApiError(error, 'validation');
        }
        setLoading(false);
    };

    // New SMS Detection Function
    const detectSMS = async () => {
        if (!message) {
            showToast('Please enter a message', 'warning');
            return;
        }
        
        setSmsLoading(true);
        try {
            const response = await axios.post(`${API_BASE}/api/detect-sms`, { message });
            
            if (response.data.error) {
                showToast(`Detection error: ${response.data.error}`, 'error');
            } else {
                setSmsResult(response.data);
                showToast('Message analyzed successfully!', 'success');
            }
        } catch (error) {
            handleApiError(error, 'SMS detection');
        }
        setSmsLoading(false);
    };

    const trainModel = async () => {
        if (!trainingFile) {
            showToast('Please select a CSV file first', 'warning');
            return;
        }
        
        setTraining(true);
        const formData = new FormData();
        formData.append('file', trainingFile);
        
        try {
            // Auto-detect which model to train based on active tab
            const endpoint = activeTab === 'sms' ? '/api/train-sms' : '/api/train';
            const response = await axios.post(`${API_BASE}${endpoint}`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            
            if (response.data.error) {
                showToast(`Training error: ${response.data.error}`, 'error');
            } else {
                showToast(`${response.data.model_type} trained successfully! Accuracy: ${(response.data.accuracy * 100).toFixed(2)}%`, 'success');
                setServerStatus('connected');
            }
        } catch (error) {
            handleApiError(error, 'training');
        }
        setTraining(false);
    };

    const handleApiError = (error, operation) => {
        if (error.code === 'NETWORK_ERROR' || error.code === 'ECONNREFUSED') {
            showToast('Cannot connect to server. Please check if the backend is running.', 'error');
            setServerStatus('disconnected');
        } else if (error.response) {
            showToast(`Server error: ${error.response.data.error || error.response.statusText}`, 'error');
        } else {
            showToast(`Error during ${operation}: ${error.message}`, 'error');
        }
    };

    const fetchHistory = async () => {
        try {
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
        
        const updatedHistory = [newHistoryItem, ...history.slice(0, 49)];
        setHistory(updatedHistory);
        localStorage.setItem('emailValidationHistory', JSON.stringify(updatedHistory));
    };

    const clearResults = () => {
        if (activeTab === 'validate') {
            setResult(null);
            setEmail('');
        } else if (activeTab === 'sms') {
            setSmsResult(null);
            setMessage('');
        }
    };

    const downloadSampleCSV = () => {
        let sampleData = '';
        
        if (activeTab === 'validate') {
            sampleData = `email,is_real
john.doe@gmail.com,1
alice.smith@yahoo.com,1
bob.johnson@hotmail.com,1
sara.wilson@outlook.com,1
mike.brown@company.com,1
emily.davis@university.edu,1
fake123@spam.com,0
random@fake-mail.xyz,0
temp@throwawaymail.com,0
test@guerrillamail.com,0
user@10minutemail.com,0
spammer@temp-mail.org,0`;
        } else if (activeTab === 'sms') {
            sampleData = `Category,Message
ham,"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
ham,"Ok lar... Joking wif u oni..."
spam,"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
ham,"U dun say so early hor... U c already then say..."
ham,"Nah I don't think he goes to usf, he lives around here though"`;
        }
        
        const blob = new Blob([sampleData], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = activeTab === 'validate' ? 'sample_email_data.csv' : 'sample_sms_data.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        showToast('Sample CSV downloaded!', 'success');
    };

    const showToast = (message, type = 'info') => {
        const existingToasts = document.querySelectorAll('.toast');
        existingToasts.forEach(toast => toast.remove());

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${getToastIcon(type)}</span>
                <span class="toast-message">${message}</span>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => toast.classList.add('show'), 100);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    };

    const getToastIcon = (type) => {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragOver(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setDragOver(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'text/csv') {
            setTrainingFile(files[0]);
            showToast('CSV file selected!', 'success');
        } else {
            showToast('Please drop a valid CSV file', 'error');
        }
    };

    const getStatusColor = (status) => {
        return status ? '#10b981' : '#ef4444';
    };

    return (
        <div className="App">
            {/* Toast Container */}
            <div className="toast-container"></div>

            {/* Header */}
            <header className="header">
                <div className="header-content">
                    <div className="logo-section">
                        <div className="logo">
                            <div className="logo-icon">🤖</div>
                            <div className="logo-text">
                                <h1>Dual AI Validator</h1>
                                <span className="tagline">Email & SMS Spam Detection</span>
                            </div>
                        </div>
                        <div className={`server-status ${serverStatus}`}>
                            <div className="status-indicator"></div>
                            {serverStatus === 'connected' ? 'Connected' : 
                             serverStatus === 'checking' ? 'Checking...' : 'Disconnected'}
                            <button onClick={checkServerStatus} className="refresh-btn" title="Refresh status">
                                🔄
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="main">
                <div className="container">
                    {/* Navigation Tabs */}
                    <nav className="tabs-nav">
                        <div className="tabs">
                            {[
                                { id: 'validate', icon: '📧', label: 'Email Validator' },
                                { id: 'sms', icon: '💬', label: 'SMS Spam Detector' },
                                { id: 'train', icon: '🏋️', label: 'Train Model' },
                                { id: 'history', icon: '📊', label: 'History' }
                            ].map(tab => (
                                <button
                                    key={tab.id}
                                    className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                                    onClick={() => setActiveTab(tab.id)}
                                >
                                    <span className="tab-icon">{tab.icon}</span>
                                    <span className="tab-label">{tab.label}</span>
                                </button>
                            ))}
                        </div>
                    </nav>

                    {/* Content Sections */}
                    <div className="content">
                        {/* Email Validation Section */}
                        {activeTab === 'validate' && (
                            <div className="section">
                                <div className="section-header">
                                    <h2>Email Validation</h2>
                                    <p>Verify email authenticity using machine learning</p>
                                </div>
                                
                                <div className="validation-card">
                                    <div className="input-group">
                                        <div className="input-wrapper">
                                            <span className="input-icon">✉️</span>
                                            <input
                                                type="email"
                                                placeholder="Enter email address to validate..."
                                                value={email}
                                                onChange={(e) => setEmail(e.target.value)}
                                                onKeyPress={(e) => e.key === 'Enter' && validateEmail()}
                                                className="email-input"
                                            />
                                        </div>
                                        <div className="action-buttons">
                                            <button 
                                                onClick={validateEmail} 
                                                disabled={loading || serverStatus !== 'connected'}
                                                className="btn btn-primary validate-btn"
                                            >
                                                {loading ? (
                                                    <>
                                                        <div className="spinner"></div>
                                                        Analyzing...
                                                    </>
                                                ) : (
                                                    <>
                                                        <span className="btn-icon">🔍</span>
                                                        Validate Email
                                                    </>
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

                                    {result && (
                                        <div className="result-section">
                                            <div className={`result-card ${result.is_real ? 'valid' : 'invalid'}`}>
                                                <div className="result-header">
                                                    <div className="result-badge">
                                                        <span className="badge-icon">
                                                            {result.is_real ? '✅' : '❌'}
                                                        </span>
                                                        <span className="badge-text">
                                                            {result.is_real ? 'Valid Email' : 'Suspicious Email'}
                                                        </span>
                                                    </div>
                                                    <div className="confidence-score">
                                                        Confidence: {(result.confidence * 100).toFixed(1)}%
                                                    </div>
                                                </div>
                                                
                                                <div className="email-display">
                                                    <code>{result.email}</code>
                                                </div>

                                                <div className="probability-metrics">
                                                    <div className="metric">
                                                        <label>Real Probability</label>
                                                        <div className="metric-bar">
                                                            <div 
                                                                className="metric-fill real"
                                                                style={{width: `${result.probability_real * 100}%`}}
                                                            >
                                                                <span className="metric-value">
                                                                    {(result.probability_real * 100).toFixed(1)}%
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="metric">
                                                        <label>Fake Probability</label>
                                                        <div className="metric-bar">
                                                            <div 
                                                                className="metric-fill fake"
                                                                style={{width: `${result.probability_fake * 100}%`}}
                                                            >
                                                                <span className="metric-value">
                                                                    {(result.probability_fake * 100).toFixed(1)}%
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* SMS Detection Section */}
                        {activeTab === 'sms' && (
                            <div className="section">
                                <div className="section-header">
                                    <h2>SMS Spam Detection</h2>
                                    <p>Detect spam messages using AI</p>
                                </div>
                                
                                <div className="validation-card">
                                    <div className="input-group">
                                        <div className="input-wrapper">
                                            <span className="input-icon">💬</span>
                                            <textarea
                                                placeholder="Enter SMS message to analyze..."
                                                value={message}
                                                onChange={(e) => setMessage(e.target.value)}
                                                className="sms-input"
                                                rows="4"
                                            />
                                        </div>
                                        <div className="action-buttons">
                                            <button 
                                                onClick={detectSMS} 
                                                disabled={smsLoading || serverStatus !== 'connected'}
                                                className="btn btn-primary validate-btn"
                                            >
                                                {smsLoading ? (
                                                    <>
                                                        <div className="spinner"></div>
                                                        Analyzing...
                                                    </>
                                                ) : (
                                                    <>
                                                        <span className="btn-icon">🔍</span>
                                                        Detect Spam
                                                    </>
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

                                    {smsResult && (
                                        <div className="result-section">
                                            <div className={`result-card ${smsResult.is_spam ? 'invalid' : 'valid'}`}>
                                                <div className="result-header">
                                                    <div className="result-badge">
                                                        <span className="badge-icon">
                                                            {smsResult.is_spam ? '🚫' : '✅'}
                                                        </span>
                                                        <span className="badge-text">
                                                            {smsResult.is_spam ? 'Spam Message' : 'Legitimate Message'}
                                                        </span>
                                                    </div>
                                                    <div className="confidence-score">
                                                        Confidence: {(smsResult.confidence * 100).toFixed(1)}%
                                                    </div>
                                                </div>
                                                
                                                <div className="message-display">
                                                    <code>{smsResult.message}</code>
                                                </div>

                                                <div className="probability-metrics">
                                                    <div className="metric">
                                                        <label>Ham Probability</label>
                                                        <div className="metric-bar">
                                                            <div 
                                                                className="metric-fill real"
                                                                style={{width: `${smsResult.probability_ham * 100}%`}}
                                                            >
                                                                <span className="metric-value">
                                                                    {(smsResult.probability_ham * 100).toFixed(1)}%
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div className="metric">
                                                        <label>Spam Probability</label>
                                                        <div className="metric-bar">
                                                            <div 
                                                                className="metric-fill fake"
                                                                style={{width: `${smsResult.probability_spam * 100}%`}}
                                                            >
                                                                <span className="metric-value">
                                                                    {(smsResult.probability_spam * 100).toFixed(1)}%
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Training Section */}
                        {activeTab === 'train' && (
                            <div className="section">
                                <div className="section-header">
                                    <h2>Model Training</h2>
                                    <p>Train the AI model with your dataset</p>
                                </div>
                                
                                <div className="training-card">
                                    <div 
                                        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${trainingFile ? 'has-file' : ''}`}
                                        onDragOver={handleDragOver}
                                        onDragLeave={handleDragLeave}
                                        onDrop={handleDrop}
                                    >
                                        <div className="upload-content">
                                            <div className="upload-icon">📁</div>
                                            <h3>Upload Training Data</h3>
                                            <p>Drag & drop your CSV file or click to browse</p>
                                            <input 
                                                type="file" 
                                                accept=".csv"
                                                onChange={(e) => {
                                                    setTrainingFile(e.target.files[0]);
                                                    showToast('CSV file selected!', 'success');
                                                }}
                                                className="file-input"
                                                id="file-upload"
                                            />
                                            <label htmlFor="file-upload" className="btn btn-outline">
                                                Choose File
                                            </label>
                                        </div>
                                        
                                        {trainingFile && (
                                            <div className="file-preview">
                                                <div className="file-info">
                                                    <span className="file-icon">📄</span>
                                                    <div className="file-details">
                                                        <span className="file-name">{trainingFile.name}</span>
                                                        <span className="file-size">
                                                            {(trainingFile.size / 1024).toFixed(2)} KB
                                                        </span>
                                                    </div>
                                                    <button 
                                                        onClick={() => setTrainingFile(null)}
                                                        className="clear-file"
                                                    >
                                                        ×
                                                    </button>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    <div className="training-actions">
                                        <button 
                                            onClick={trainModel} 
                                            disabled={!trainingFile || training || serverStatus !== 'connected'}
                                            className="btn btn-primary train-btn"
                                        >
                                            {training ? (
                                                <>
                                                    <div className="spinner"></div>
                                                    Training in Progress...
                                                </>
                                            ) : (
                                                <>
                                                    <span className="btn-icon">🚀</span>
                                                    Train Model
                                                </>
                                            )}
                                        </button>
                                        
                                        <button 
                                            onClick={downloadSampleCSV}
                                            className="btn btn-secondary"
                                        >
                                            <span className="btn-icon">📥</span>
                                            Download Sample CSV
                                        </button>
                                    </div>

                                    <div className="format-guide">
                                        <h4>CSV Format Requirements</h4>
                                        {activeTab === 'train' && (
                                            <>
                                                <div className="format-options">
                                                    <div className="format-option">
                                                        <h5>For Email Validation:</h5>
                                                        <div className="code-example">
                                                            <pre>{`email,is_real\njohn.doe@domain.com,1\nfake.email@spam.com,0\n...`}</pre>
                                                        </div>
                                                    </div>
                                                    <div className="format-option">
                                                        <h5>For SMS Spam Detection:</h5>
                                                        <div className="code-example">
                                                            <pre>{`Category,Message\nham,"Hello, how are you?"\nspam,"Free prize! Click here!"\n...`}</pre>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div className="format-notes">
                                                    <div className="note">
                                                        <span className="note-badge real">1/ham</span>
                                                        <span>Real email or Legitimate message</span>
                                                    </div>
                                                    <div className="note">
                                                        <span className="note-badge fake">0/spam</span>
                                                        <span>Fake email or Spam message</span>
                                                    </div>
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* History Section */}
                        {activeTab === 'history' && (
                            <div className="section">
                                <div className="section-header">
                                    <h2>Validation History</h2>
                                    <p>Recent email validation results and analysis</p>
                                </div>
                                
                                <div className="history-card">
                                    {history.length === 0 ? (
                                        <div className="empty-state">
                                            <div className="empty-icon">📊</div>
                                            <h3>No Validation History</h3>
                                            <p>Start validating emails to see your history here</p>
                                            <button 
                                                onClick={() => setActiveTab('validate')}
                                                className="btn btn-primary"
                                            >
                                                Validate First Email
                                            </button>
                                        </div>
                                    ) : (
                                        <>
                                            <div className="history-stats">
                                                <div className="stat">
                                                    <span className="stat-value">{history.length}</span>
                                                    <span className="stat-label">Total Validations</span>
                                                </div>
                                                <div className="stat">
                                                    <span className="stat-value">
                                                        {history.filter(h => h.isReal).length}
                                                    </span>
                                                    <span className="stat-label">Valid Emails</span>
                                                </div>
                                                <div className="stat">
                                                    <span className="stat-value">
                                                        {history.filter(h => !h.isReal).length}
                                                    </span>
                                                    <span className="stat-label">Suspicious Emails</span>
                                                </div>
                                            </div>
                                            
                                            <div className="history-list">
                                                {history.map((item, index) => (
                                                    <div key={index} className="history-item">
                                                        <div className="item-main">
                                                            <div className="email-address">
                                                                <code>{item.email}</code>
                                                            </div>
                                                            <div className="item-meta">
                                                                <span className="timestamp">
                                                                    {new Date(item.timestamp).toLocaleString()}
                                                                </span>
                                                            </div>
                                                        </div>
                                                        <div className="item-result">
                                                            <div 
                                                                className={`status-indicator ${item.isReal ? 'valid' : 'invalid'}`}
                                                                style={{ 
                                                                    backgroundColor: getStatusColor(item.isReal) 
                                                                }}
                                                            ></div>
                                                            <span className={`status-text ${item.isReal ? 'valid' : 'invalid'}`}>
                                                                {item.isReal ? 'Valid' : 'Suspicious'}
                                                            </span>
                                                            <div className="confidence">
                                                                {(item.confidence * 100).toFixed(1)}%
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                            
                                            <div className="history-actions">
                                                <button 
                                                    onClick={() => {
                                                        localStorage.removeItem('emailValidationHistory');
                                                        setHistory([]);
                                                        showToast('History cleared!', 'success');
                                                    }}
                                                    className="btn btn-outline"
                                                >
                                                    Clear History
                                                </button>
                                            </div>
                                        </>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="footer">
                <div className="container">
                    <div className="footer-content">
                        <p>&copy; 2024 Dual AI Validator. Email & SMS spam detection system.</p>
                        <div className="footer-status">
                            <span>Backend: </span>
                            <span className={`status ${serverStatus}`}>
                                {serverStatus === 'connected' ? '🟢 Online' : '🔴 Offline'}
                            </span>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
}

export default App;