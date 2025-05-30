# AI-Powered-Public-Speaking-Training Platform

# Overview
This is a web-based platform built with Flask that helps users improve their public speaking skills through three interactive exercises: Conductor, Triple Step, and Rapid Fire Analogies. It uses advanced speech analysis techniques, Whisper for transcription, MFCC features for audio processing, and a local LLM (like LLaMA) for personalized feedback.
# 🚀 Features
User Authentication – Sign up and log in securely using MySQL.
Conductor Mode – Emotion analysis using a pre-trained audio transformer model and comparison with expected scores.
Triple Step Mode – Speech evaluation with distractor words, filler words, and grammar checks.
Rapid Fire Analogies Mode – Transcription and feedback on rapid-fire analogy questions.
LLM-Based Feedback – LLaMA generates context-aware and constructive suggestions.
Audio Transcription – Powered by Whisper.
Speech Analysis – Includes MFCC extraction, filler word detection, and emotional classification.

# 🛠️ Tech Stack
Backend: Python (Flask)
Database: MySQL
Audio Processing: Whisper, librosa, torchaudio
Machine Learning: Transformers, HuggingFace, OpenAI Whisper, LLaMA via Langchain
Frontend: HTML, CSS (via templates)
File Storage: Local file system for audio files

# 📁 Project Directory Structure

```bash
├── app.py                        # Main Flask application
├── templates/                   # HTML templates for UI
│   ├── login.html
│   ├── choose.html
│   ├── conductor.html
│   ├── triple_step.html
│   └── analogy.html
├── uploads/                     # Folder to store uploaded audio files
├── speech_analysis.py           # Logic for Triple Step speech analysis
├── conductor.py                 # Logic for emotion analysis and scoring
├── analogy.py                   # Rapid Fire analogy functionality
├── llm_feedback.py              # LLM (LLaMA) based feedback generation
├── requirements.txt             # List of Python dependencies
└── README.md                    # Project documentation
```

# 🧪 How to Run the App

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the application:

```bash
python app.py
```

> ⚠️ Make sure MySQL server is running and database `speechassistantdb` is set up.

# Default Config (Change in Production)
Secret Key: your_secret_key

MySQL:

Host: localhost

User: root

Password: ahalya_19

DB: speechassistantdb

> ⚠️ Change credentials and secret key in production environments!

# 📌 Notes
Ensure FFmpeg is installed for Whisper to work properly.
Ollama must be running locally for feedback generation using LLaMA.
This project supports modular analysis and can be extended with more exercises or metrics.

# 🧠 Credits
Developed as part of an academic AI project to enhance public speaking through intelligent analysis and feedback.





