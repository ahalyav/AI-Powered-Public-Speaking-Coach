import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import nltk
from nltk.corpus import stopwords
import language_tool_python
import whisper

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Initialize grammar checker
tool = language_tool_python.LanguageTool("en-US")

# Function to transcribe speech using Whisper
def transcribe_audio_with_whisper_triple(input_file):
    model = whisper.load_model("base")
    result = model.transcribe(input_file, verbose=False)
    transcription = result["text"]
    print("\nTranscription:", transcription)
    return transcription

# Function to analyze speech
def analyze_speech_triple(text, distractor_words):
    words = text.split()
    stop_word_count = sum(1 for word in words if word.lower() in stop_words)
    filler_words = {"um", "uh", "like", "you know", "so", "actually", "basically"}
    filler_word_count = sum(1 for word in words if word.lower() in filler_words)
    grammar_errors = tool.check(text)
    distractor_count = sum(1 for word in words if word.lower() in distractor_words)
    
    analysis_results = {
        "total_words": len(words),
        "stop_word_count": stop_word_count,
        "filler_word_count": filler_word_count,
        "distractor_count": distractor_count,
        "grammar_mistakes": len(grammar_errors),
        "grammar_suggestions": [
            {"rule": error.ruleId, "message": error.message, "suggestions": error.replacements}
            for error in grammar_errors[:5]
        ]
    }

    print("\n=== Speech Analysis ===")
    print(f"Total words: {analysis_results['total_words']}")
    print(f"Stop words: {analysis_results['stop_word_count']} (Try reducing them for clarity)")
    print(f"Filler words: {analysis_results['filler_word_count']} (Avoid unnecessary pauses)")
    print(f"Distractor words used: {analysis_results['distractor_count']} (Lower is better)")
    print(f"Grammar Mistakes: {analysis_results['grammar_mistakes']} (Suggestions below)")
    
    if grammar_errors:
        for i, error in enumerate(grammar_errors[:5], 1):
            print(f"{i}. {error.ruleId}: {error.message} (Suggestion: {error.replacements})")

    return analysis_results

# Function to extract MFCC features
def extract_mfcc_features_triple(audio_file, n_mfcc=13):
    y, sr = librosa.load(audio_file, sr=44100)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) == 0:
        print("\nNo valid speech detected. Please try again.")
        return np.zeros(n_mfcc)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(label="MFCC Coefficients")
    plt.title("MFCC Features of Speech")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.show()
    
    return mfcc_mean
