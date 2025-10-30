# 🎓 University AI Chatbot

This is a simple, intent-based AI chatbot built for a university project. It uses a Machine Learning model (LinearSVC) trained on a custom dataset to understand and respond to user queries about admissions, courses, fees, and more.

The frontend is built with Streamlit, providing a clean, real-time chat interface.

## 🚀 Features

* **Intent Recognition**: Understands user questions related to 11 different topics (intents).
* **Dynamic Responses**: Provides a random, natural-sounding response for each recognized intent.
* **Web-Based UI**: A simple and responsive chat interface built with Streamlit.
* **Suggestion Buttons**: A "quick-start" guide for new users to see what the bot can do.
* **Debug Mode**: Shows the predicted intent (`tag`) in the UI for easy testing.

---

## 🛠️ Tech Stack

* **Backend**: Python
* **ML Model**: Scikit-learn (`LinearSVC` + `TfidfVectorizer` Pipeline)
* **NLP**: NLTK (for tokenizing and lemmatizing)
* **Frontend**: Streamlit
* **Data**: `intents.json` (custom dataset)

---

## 🏃‍♂️ How to Run This Project Locally

### 1. Prerequisites

* Python 3.8 or newer
* `pip` and `venv` (standard Python libraries)

### 2. Setup

Clone the repository (or use your local folder) and navigate into it:

```bash
git clone [https://github.com/rishabh794/university-chatbot.git](https://github.com/rishabh794/university-chatbot.git)
cd university-chatbot
