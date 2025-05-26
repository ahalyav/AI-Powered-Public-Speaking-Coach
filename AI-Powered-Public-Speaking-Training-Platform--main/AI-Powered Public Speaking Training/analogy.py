import time
import pyaudio
import wave
import whisper
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama

# Load the LLaMA model (Mistral via Ollama)
llm = Ollama(model="llama3.2")

def generate_questions(duration):
    """Generate AI-generated speaking prompts based on user-selected duration."""
    template = """The user's selected time duration is {duration} seconds. 
    Based on the selected time, generate a corresponding number of questions
    '(1 question per 5 seconds)'.

    Each question should provide 'exactly two starting words'. 
    Ensure the starting words are varied and engaging. Format each question as:

    Example:
    'Virtual reality enhances ?'
    'Machine learning optimizes ?'
    'Artificial intelligence is ?'

    Do not include these examples in the user’s questions.
    The questions should be generic and understandable by everyone.
    Generate only the required number of questions.
    No introduction should be given, only the questions should be generated.
    Here are the question strictly should not be given"""

    formatted_prompt = template.format(duration=duration)
    analogy_response = llm.invoke(formatted_prompt)
    return [line.strip() for line in analogy_response.split("\n") if line.strip()]

def record_audio(output_file, record_seconds=10, sample_rate=44100, chunk_size=1024):
    """Record audio for the given duration."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("\n🎤 Recording started... Answer the displayed questions!")

    frames = []
    start_time = time.time()

    while time.time() - start_time < record_seconds:
        data = stream.read(chunk_size)
        frames.append(data)

    print("\n🎤 Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

def transcribe_audio_with_whisper(input_file):
    """Transcribe recorded speech using Whisper AI."""
    model = whisper.load_model("base")
    result = model.transcribe(input_file, verbose=False)
    transcription = result["text"]
    print("\n📜 Transcription:", transcription)
    return transcription

def analyze_speech(text):
    """Analyze speech by counting filler words."""
    words = text.split()
    filler_words = {"um", "uh", "like", "you know", "so", "actually", "basically"}
    filler_word_count = sum(1 for word in words if word.lower() in filler_words)
    print("\n=== Speech Analysis ===")
    print(f"Total words: {len(words)}")
    print(f"Filler words: {filler_word_count} (Avoid unnecessary pauses)")
    return filler_word_count

def extract_mfcc_features(audio_file, n_mfcc=13):
    """Extract and display MFCC features from the audio."""
    y, sr = librosa.load(audio_file, sr=44100)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) == 0:
        print("\n⚠️ No valid speech detected. Please try again.")
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

def get_llama_feedback_analogy(generated_analogies, transcription, mfcc_features, filler_count):
    """Generate AI feedback based on speech performance."""
    mfcc_summary = ", ".join([f"{feat:.2f}" for feat in mfcc_features])
    
    feedback_prompt = f"""
    Evaluate the following response based on relevance to the generated analogy, pronunciation, creativity, and quality
    Additionally, consider the MFCC features to assess the quality of the speaker's speech.
    Also, give a score based on reasoning for the analogy.
    GIVE THE SPEAKER A SCORE OUT OF 100.
    Give the dcoring based on how creatively he completes the sentence.
    give score based on the number of questions he answers correct

    **Questions:** 
    {chr(10).join(generated_analogies)}

    **Transcription:** {transcription}

    **Filler count:** {filler_count}

    **MFCC Features (mean values):** {mfcc_summary}

    Feedback should include:
    - Relevance to the asked question(The speaker just repeating the question is not enough, He/she has to complete the sentence)
    - Creativity(how creatively he completes the question) 
    - Clarity(clarity in his speech)(assign a small part of the overall score for this)
    - Overall effectiveness
    """
    
    feedback_response = llm.invoke(feedback_prompt)
    print("\n=== LLaMA AI Feedback ===")
    print(feedback_response)
    clean_output = feedback_response.replace("*", "")

    return clean_output

def run_speech_analysis():
    """Run the entire speech evaluation process."""
    user_duration = int(input("Enter the duration in seconds (multiple of 5): ").strip())
    generated_analogies = generate_questions(user_duration)

    print("\nGenerated Analogies:")
    for analogy in generated_analogies:
        print(f"🔹 {analogy}")

    audio_file = "output.wav"
    record_audio(audio_file, record_seconds=user_duration)
    transcription = transcribe_audio_with_whisper(audio_file)
    filler_count = analyze_speech(transcription)
    mfcc_features = extract_mfcc_features(audio_file)

    get_llama_feedback_analogy(generated_analogies, transcription, mfcc_features, filler_count)

if __name__ == "__main__":
    run_speech_analysis()
