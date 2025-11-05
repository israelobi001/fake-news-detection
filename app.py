from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import time

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)

class NewsPredictor:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open('models/model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            print("✓ Model loaded successfully!")
        except FileNotFoundError:
            print("WARNING: Model files not found. Please train the model first.")
            print("Run 'python train_model.py' to train the model.")
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def get_top_words(self, text, top_n=5):
        """Get top contributing words for explanation"""
        cleaned = self.preprocess_text(text)
        words = cleaned.split()
        
        # Get unique words and their frequency
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, freq in sorted_words[:top_n]]
        
        return top_words
    
    def generate_explanation(self, text, is_fake, confidence, top_words):
        """Generate human-readable explanation"""
        if confidence < 60:
            return "Model is uncertain - text may be ambiguous or too short for reliable classification"
        
        if is_fake:
            if confidence > 85:
                explanation = f"Strong indicators of misinformation detected. Key terms: {', '.join(top_words[:3])}"
            else:
                explanation = f"Potential misinformation patterns identified. Analysis based on: {', '.join(top_words[:3])}"
        else:
            if confidence > 85:
                explanation = f"Language appears credible and informational. Key terms: {', '.join(top_words[:3])}"
            else:
                explanation = f"Likely authentic but with some uncertainty. Based on: {', '.join(top_words[:3])}"
        
        return explanation
    
    def predict(self, text):
        """Predict if news is fake or real"""
        if not self.model or not self.vectorizer:
            return {
                'error': 'Model not loaded. Please train the model first.',
                'prediction': None,
                'confidence': None
            }
        
        if not text or len(text.strip()) < 10:
            return {
                'error': 'Please provide a longer text (at least 10 characters).',
                'prediction': None,
                'confidence': None
            }
        
        try:
            # Preprocess
            cleaned = self.preprocess_text(text)
            
            if not cleaned or len(cleaned.split()) < 3:
                return {
                    'error': 'Text is too short after preprocessing. Please provide more content.',
                    'prediction': None,
                    'confidence': None
                }
            
            # Vectorize
            vectorized = self.vectorizer.transform([cleaned])
            
            # Predict
            prediction = self.model.predict(vectorized)[0]
            probabilities = self.model.predict_proba(vectorized)[0]
            
            # Get probabilities
            real_prob = probabilities[0] * 100
            fake_prob = probabilities[1] * 100
            confidence = max(real_prob, fake_prob)
            
            # Get top contributing words
            top_words = self.get_top_words(text)
            
            # Generate explanation
            is_fake = bool(prediction == 1)
            explanation = self.generate_explanation(text, is_fake, confidence, top_words)
            
            return {
                'prediction': 'Fake News' if is_fake else 'Real News',
                'confidence': float(confidence),
                'is_fake': is_fake,
                'real_probability': float(real_prob),
                'fake_probability': float(fake_prob),
                'explanation': explanation,
                'top_words': top_words,
                'error': None
            }
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'prediction': None,
                'confidence': None
            }

# Initialize predictor
predictor = NewsPredictor()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'success': False
            }), 400
        
        # Get prediction
        result = predictor.predict(text)
        
        if result.get('error'):
            return jsonify({
                'error': result['error'],
                'success': False
            }), 400
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'is_fake': result['is_fake'],
            'real_probability': result['real_probability'],
            'fake_probability': result['fake_probability'],
            'explanation': result['explanation'],
            'top_words': result.get('top_words', []),
            'processing_time_ms': round(processing_time, 2)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = predictor.model is not None and predictor.vectorizer is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model_not_loaded',
        'model_loaded': model_loaded
    })

@app.route('/stats')
def stats():
    """Get model statistics"""
    if not predictor.model or not predictor.vectorizer:
        return jsonify({
            'error': 'Model not loaded',
            'loaded': False
        })
    
    # Get model info
    from sklearn.linear_model import LogisticRegression
    
    model_type = type(predictor.model).__name__
    vocab_size = len(predictor.vectorizer.vocabulary_)
    
    return jsonify({
        'loaded': True,
        'model_type': model_type,
        'vocabulary_size': vocab_size,
        'features': predictor.vectorizer.max_features,
        'ngram_range': predictor.vectorizer.ngram_range
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("\n" + "="*60)
    print("FAKE NEWS DETECTION SYSTEM")
    print("="*60)
    print("\nStarting Flask application...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)