import pyaudio
import wave
import librosa
import torch
import numpy as np
import random
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from langchain_community.llms import Ollama

# Load the LLaMA model (Mistral via Ollama)
llm = Ollama(model="llama3.2")

# Generate a topic
def generate_topic():
    topic_prompt = ("Generate a general topic for a speaker to talk about speaker. "
                    "Give only the topic. Do not include any extra words.")
    topic_response = llm.invoke(topic_prompt)
    generated_topic = topic_response.strip()
    print("\nGenerated Topic:", generated_topic)
    return generated_topic

# Get user input for duration
def get_recording_duration():
    while True:
        try:
            duration = int(input("Enter recording duration (60-120 seconds): "))
            if 60 <= duration <= 120:
                return duration
            else:
                print("Please enter a value between 60 and 120.")
        except ValueError:
            print("Invalid input. Please enter an integer between 60 and 120.")

# Generate random target scores for each 10s segment
def generate_expected_scores(duration):
    num_segments = duration // 10  # Total number of 10-second chunks
    expected_scores = [random.randint(1, 10) for _ in range(num_segments)]
    return expected_scores

# Record audio
def record_audio(filename, duration, sample_rate=16000):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, frames_per_buffer=1024, input=True)

    print("\nRecording... Speak with the expected score for each segment!\n")
    frames = []

    num_segments = duration // 10
    for i in range(num_segments):
        print(f"Segment {i+1}: Speak with **Score: {expected_scores[i]}** for the next 10 seconds.")
        for _ in range(int(sample_rate / 1024 * 10)):  # Capture for 10 seconds
            frames.append(stream.read(1024))

    print("\nRecording complete.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

# Split audio into 10-second segments
def split_audio(audio_path, chunk_duration=10):
    audio_array, sr = librosa.load(audio_path, sr=16000)
    chunk_length = sr * chunk_duration  # Number of samples in 10s
    num_chunks = len(audio_array) // chunk_length  # Total number of 10s segments
    
    chunks = [audio_array[i * chunk_length: (i + 1) * chunk_length] for i in range(num_chunks)]
    return chunks, sr

# Predict emotion per chunk
def predict_emotion_per_chunk(chunks, sr, model, feature_extractor, id2label, expected_scores):
    emotion_scores = {
        "neutral": 5,
        "sad": 1,
        "Surprised": 7,
        "disgust": 3,
        "fearful": 6,
        "angry": 8,
        "happy": 10
    }
    score_to_emotion = {v: k for k, v in emotion_scores.items()}
    results = {}

    for i, chunk in enumerate(chunks):
        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt", truncation=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1).squeeze().tolist()

        # Map to emotions
        emotion_probs = {id2label[i]: prob for i, prob in enumerate(probabilities)}
        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)

        predicted_emotion = sorted_emotions[0][0]  # Emotion with the highest probability
        predicted_score = emotion_scores.get(predicted_emotion, 5)
        expected_score = expected_scores[i]

        # Store results
        results[f"segment_{i+1}"] = {
            "expected": expected_score,
            "detected": predicted_emotion,
            "predicted_score": predicted_score,
            "probability": sorted_emotions[0][1]
        }

        print(f"\nSegment {i+1}:")
        print(f"Expected Emotion: {score_to_emotion.get(expected_score, 'neutral').upper()} (Score: {expected_score})")
        print(f"Predicted Emotion: {predicted_emotion.upper()} (Score: {predicted_score})")
        print(f"Confidence: {sorted_emotions[0][1]:.4f}")

    return results

# Calculate weighted accuracy
def calculate_weighted_accuracy(results):
    total_possible_score = len(results) * 10  # Max score per segment is 10
    total_weighted_score = 0  

    for res in results.values():
        expected = res["expected"]
        predicted = res["predicted_score"]
        diff = abs(expected - predicted)

        # Assign weighted scores based on difference
        if diff == 0:
            weighted_score = 10  # Perfect match
        elif diff == 1:
            weighted_score = 9
        elif diff == 2:
            weighted_score = 8
        elif diff == 3:
            weighted_score = 6
        elif diff > 5:
            weighted_score = 0  # Way off
        else:
            weighted_score = 3  # Moderate miss
        
        total_weighted_score += weighted_score

    accuracy = (total_weighted_score / total_possible_score) * 100
    return round(accuracy, 2)

# Generate feedback based on emotion recognition results
def get_llama_feedback(topic, emotion_results, accuracy_score):
    feedback_prompt = f"""
    Evaluate the following speech based on the emotion recognition results and accuracy score.
    Provide feedback on how the speaker can improve their emotional expression.

    **Topic:** {topic}

    **Emotion Recognition Results:**
    {emotion_results}

    **Accuracy Score:** {accuracy_score}%

    Feedback should include:
    - How well the speaker expressed the expected emotions
    - Suggestions for improving emotional expression
    - Overall effectiveness of the speech in conveying emotions
    """
    
    feedback_response = llm.invoke(feedback_prompt)
    print("\n=== LLaMA AI Feedback ===")
    print(feedback_response)
    clean_output = feedback_response.replace("*", "")

    return clean_output

# Main function to run the entire process
def main():
    # Emotion Scores Mapping
    global emotion_scores, score_to_emotion, expected_scores
    emotion_scores = {
        "neutral": 5,
        "sad": 1,
        "Surprised": 7,
        "disgust": 3,
        "fearful": 6,
        "angry": 8,
        "happy": 10
    }
    score_to_emotion = {v: k for k, v in emotion_scores.items()}

    # Generate a topic for the speaker to talk about
    topic = generate_topic()

    # Get user input for duration
    duration = get_recording_duration()

    # Generate random target scores for each 10s segment
    expected_scores = generate_expected_scores(duration)
    expected_emotions = [score_to_emotion.get(score, "neutral") for score in expected_scores]

    # Print the randomly generated expected scores
    print(f"\nGenerated expected scores per segment: {expected_scores}")

    # Record audio using user input
    filename = "recorded_audio.wav"
    record_audio(filename, duration)

    # Load model
    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
    id2label = model.config.id2label

    # Split the recorded audio into 10-second segments
    audio_chunks, sampling_rate = split_audio(filename)

    # Run emotion detection per segment
    emotion_results = predict_emotion_per_chunk(audio_chunks, sampling_rate, model, feature_extractor, id2label, expected_scores)

    # Compute final assessment score
    accuracy_score = calculate_weighted_accuracy(emotion_results)
    print(f"\nFinal Accuracy Score: {accuracy_score}%")

    # Generate feedback based on emotion recognition results
    get_llama_feedback(topic, emotion_results, accuracy_score)

if __name__ == "__main__":
    main()