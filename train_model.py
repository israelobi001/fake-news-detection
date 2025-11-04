import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
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
    
    def load_isot_dataset(self, fake_path='Fake.csv', true_path='True.csv'):
        """Load ISOT dataset from two separate CSV files"""
        print("="*60)
        print("Loading ISOT Fake News Dataset...")
        print("="*60)
        
        # Load fake news
        print(f"\nLoading fake news from: {fake_path}")
        fake_df = pd.read_csv(fake_path)
        fake_df['label'] = 1  # 1 = Fake
        print(f"✓ Loaded {len(fake_df)} fake news articles")
        
        # Load true news
        print(f"\nLoading true news from: {true_path}")
        true_df = pd.read_csv(true_path)
        true_df['label'] = 0  # 0 = Real/True
        print(f"✓ Loaded {len(true_df)} true news articles")
        
        # Combine datasets
        print("\nCombining datasets...")
        df = pd.concat([fake_df, true_df], ignore_index=True)
        print(f"✓ Total articles: {len(df)}")
        
        # Check columns
        print(f"\nDataset columns: {list(df.columns)}")
        
        # ISOT dataset usually has: title, text, subject, date
        # We'll combine title and text for better accuracy
        if 'title' in df.columns and 'text' in df.columns:
            print("\nCombining 'title' and 'text' columns...")
            df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            text_column = 'combined_text'
        elif 'text' in df.columns:
            text_column = 'text'
        elif 'title' in df.columns:
            text_column = 'title'
        else:
            text_column = df.columns[0]
        
        print(f"Using '{text_column}' for training")
        
        # Remove empty rows
        df = df.dropna(subset=[text_column, 'label'])
        df = df[df[text_column].str.strip() != '']
        
        print(f"\nAfter cleaning: {len(df)} articles")
        print(f"  - Fake news: {len(df[df['label'] == 1])}")
        print(f"  - Real news: {len(df[df['label'] == 0])}")
        
        # Preprocess text
        print("\nPreprocessing text (this may take a few minutes)...")
        df['cleaned_text'] = df[text_column].apply(self.preprocess_text)
        
        # Remove empty cleaned text
        df = df[df['cleaned_text'].str.strip() != '']
        
        print(f"✓ Preprocessing complete! Final dataset: {len(df)} articles")
        
        return df['cleaned_text'], df['label']
    
    def train(self, X, y, model_type='logistic'):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        print(f"\nSplitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  - Training set: {len(X_train)} articles")
        print(f"  - Test set: {len(X_test)} articles")
        
        print("\nVectorizing text with TF-IDF...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print(f"✓ Feature matrix shape: {X_train_vec.shape}")
        
        print(f"\nTraining {model_type.upper()} model...")
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        self.model.fit(X_train_vec, y_train)
        print("✓ Model training complete!")
        
        # Evaluate
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        y_train_pred = self.model.predict(X_train_vec)
        y_test_pred = self.model.predict(X_test_vec)
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\n📊 ACCURACY SCORES:")
        print(f"  - Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        print("\n📋 DETAILED CLASSIFICATION REPORT:")
        print("-"*60)
        print(classification_report(y_test, y_test_pred, 
                                   target_names=['Real News', 'Fake News'],
                                   digits=4))
        
        print("📊 CONFUSION MATRIX:")
        print("-"*60)
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"                  Predicted Real  Predicted Fake")
        print(f"Actual Real       {cm[0][0]:>14}  {cm[0][1]:>14}")
        print(f"Actual Fake       {cm[1][0]:>14}  {cm[1][1]:>14}")
        
        return test_accuracy
    
    def save_model(self, vectorizer_path='models/vectorizer.pkl', model_path='models/model.pkl'):
        """Save trained model and vectorizer"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        import os
        os.makedirs('models', exist_ok=True)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✓ Vectorizer saved to: {vectorizer_path}")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to: {model_path}")
    
    def test_predictions(self):
        """Test the model with sample predictions"""
        print("\n" + "="*60)
        print("TESTING SAMPLE PREDICTIONS")
        print("="*60)
        
        test_samples = [
            "Scientists discover miracle cure that eliminates all diseases overnight",
            "Government announces new infrastructure development project",
            "Breaking news: Aliens have invaded Earth and taken over the White House",
            "Stock market shows steady growth in the technology sector",
            "This one weird trick will make you rich instantly doctors hate it"
        ]
        
        for i, text in enumerate(test_samples, 1):
            cleaned = self.preprocess_text(text)
            vectorized = self.vectorizer.transform([cleaned])
            prediction = self.model.predict(vectorized)[0]
            probability = self.model.predict_proba(vectorized)[0]
            
            result = 'FAKE' if prediction == 1 else 'REAL'
            confidence = max(probability) * 100
            
            print(f"\n{i}. Text: {text[:60]}...")
            print(f"   Prediction: {result} (Confidence: {confidence:.2f}%)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("ISOT Dataset (Fake.csv + True.csv)")
    print("="*60)
    
    try:
        # Initialize detector
        detector = FakeNewsDetector()
        
        # Load ISOT dataset (Fake.csv and True.csv)
        X, y = detector.load_isot_dataset(fake_path='Fake.csv', true_path='True.csv')
        
        # Train model
        # Options: 'logistic', 'naive_bayes', 'random_forest'
        accuracy = detector.train(X, y, model_type='logistic')
        
        # Save model
        detector.save_model()
        
        # Test with sample predictions
        detector.test_predictions()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nYour model is ready! Run 'python app.py' to start the web application.")
        print("\n" + "="*60)
        
    except FileNotFoundError as e:
        print("\n" + "="*60)
        print("❌ ERROR: Dataset files not found!")
        print("="*60)
        print("\nPlease make sure you have both files in the same directory:")
        print("  1. Fake.csv (fake news articles)")
        print("  2. True.csv (true news articles)")
        print("\nDownload from: https://www.uvic.ca/engineering/ece/isot/datasets/")
        print("="*60)
    
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ ERROR: {str(e)}")
        print("="*60)