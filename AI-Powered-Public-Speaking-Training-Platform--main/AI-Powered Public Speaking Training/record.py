import pyaudio
import wave

def record_audio(output_file, record_seconds=5, sample_rate=44100, chunk_size=1024):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording...")

    frames = []

    # Record for the specified number of seconds
    for _ in range(int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a .wav file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

# Example usage
record_audio("output.wav")


def play_audio(input_file):
    # Open the .wav file
    with wave.open(input_file, 'rb') as wf:
        audio = pyaudio.PyAudio()

        # Open stream
        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

        # Read and play audio data
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Example usage
play_audio("output.wav")
