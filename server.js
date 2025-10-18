const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const axios = require('axios');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect('mongodb://localhost:27017/email_validator', {
    useNewUrlParser: true,
    useUnifiedTopology: true
});

// Schema for storing validation history
const validationSchema = new mongoose.Schema({
    email: String,
    isReal: Boolean,
    confidence: Number,
    timestamp: { type: Date, default: Date.now }
});

const Validation = mongoose.model('Validation', validationSchema);

// Routes
app.post('/api/validate-email', async (req, res) => {
    try {
        const { email } = req.body;
        
        // Call Python ML API
        const response = await axios.post('http://localhost:5000/api/validate', {
            email: email
        });
        
        // Save to MongoDB
        const validationRecord = new Validation({
            email: response.data.email,
            isReal: response.data.is_real,
            confidence: response.data.confidence
        });
        await validationRecord.save();
        
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/validation-history', async (req, res) => {
    try {
        const history = await Validation.find().sort({ timestamp: -1 }).limit(50);
        res.json(history);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});