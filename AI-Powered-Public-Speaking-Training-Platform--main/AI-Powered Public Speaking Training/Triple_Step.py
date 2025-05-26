import time
import threading
import random
import pyaudio
import wave
from speech_analysis import transcribe_audio_with_whisper_triple, analyze_speech_triple, extract_mfcc_features_triple
from llm_feedback import generate_topic_triple, generate_distractor_words, get_llama_feedback_triple

# Function to display one random distractor word at equal intervals
def display_distractors(distractor_words, duration, interval=5):
    start_time = time.time()
    while time.time() - start_time < duration:
        word = random.choice(distractor_words)
        print(f"\nDistractor Word: {word}")
        time.sleep(interval)

# Function to record audio
def record_audio(output_file, record_seconds=30, sample_rate=44100, chunk_size=1024):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    print("\nRecording... Speak about the given topic.")

    distractor_thread = threading.Thread(target=display_distractors, args=(distractor_words, record_seconds, 8))
    distractor_thread.start()
    
    frames = []
    for _ in range(int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)
    print("\nFinished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

# Main execution
if __name__ == "__main__":
    audio_file = "output.wav"

    # Generate topic and distractor words
    generated_topic = generate_topic_triple()
    distractor_words = generate_distractor_words(generated_topic)

    # Record and analyze speech
    record_audio(audio_file, record_seconds=40)
    transcription = transcribe_audio_with_whisper_triple(audio_file)
    distractor_count = analyze_speech_triple(transcription, distractor_words)
    mfcc_features = extract_mfcc_features_triple(audio_file)

    # Get AI feedback
    get_llama_feedback_triple(generated_topic, transcription, mfcc_features, distractor_count)
