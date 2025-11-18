import json
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score 

# --- 1. NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK data (punkt, wordnet, omw-1.4, punkt_tab)...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("Download complete.")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# --- 2. Custom Text Preprocessing Function ---
def preprocess_text(text):
    """
    Tokenizes and lemmatizes the input text.
    """
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# --- 3. Load and Prepare Data ---
print("Loading intents data...")
with open('intents.json', 'r') as f:
    data = json.load(f)

patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

print(f"Loaded {len(patterns)} patterns and {len(tags)} tags.")

# --- 4. Create and Train the UPGRADED ML Pipeline ---
print("Creating advanced ML pipeline (TF-IDF + LinearSVC)...")

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        tokenizer=preprocess_text,
        ngram_range=(1, 3),      
        max_features=5000        
    )),
    
    ('classifier', LinearSVC(    
        random_state=42,
        C=1.0,                   
        max_iter=3000           
    ))
])

# --- 5. Train the Model ---
print("Training the new model...")
pipeline.fit(patterns, tags)

# --- 6. Evaluate the Model 
print("Evaluating model on training data...")
predictions = pipeline.predict(patterns)
accuracy = accuracy_score(tags, predictions) 
print(f"\nModel Accuracy (on training data): {accuracy * 100:.2f}%")
if accuracy > 0.98:
    print("This is a fantastic result! The model has learned the data.")
else:
    print("Model accuracy is okay, but could be better. Check for errors.")

# --- 7. Save the Model ---
joblib.dump(pipeline, 'chatbot_model.joblib')
print(f"\nModel training complete!")
print("NEW advanced model saved as 'chatbot_model.joblib'")
