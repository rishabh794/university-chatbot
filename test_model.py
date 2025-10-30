import joblib
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Load all the helpers from app.py ---
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# --- Start of the test script ---
print("Loading model and intents...")
pipeline = joblib.load('chatbot_model.joblib')
with open('intents.json', 'r') as f:
    data = json.load(f)

print("Starting test...")
correct_predictions = 0
total_patterns = 0
errors = []

# Loop through every intent and every pattern
for intent in data['intents']:
    expected_tag = intent['tag']
    for pattern in intent['patterns']:
        total_patterns += 1
        
        # Get the model's prediction
        predicted_tag = pipeline.predict([pattern])[0]
        
        if predicted_tag == expected_tag:
            correct_predictions += 1
        else:
            # Log the error
            errors.append(
                f"  Pattern: '{pattern}'\n"
                f"  Expected: '{expected_tag}'\n"
                f"  Got: '{predicted_tag}'\n"
            )

# --- Print the Report ---
print("\n--- Test Report ---")
accuracy = (correct_predictions / total_patterns) * 100
print(f"Accuracy: {accuracy:.2f}% ({correct_predictions} / {total_patterns} correct)")

if errors:
    print("\n--- Mismatches Found ---")
    for error in errors:
        print(error)
else:
    print("\nAll known patterns passed! ✅")