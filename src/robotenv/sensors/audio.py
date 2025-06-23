import wave

import numpy as np
import pyaudio


class Audio(object):
    def __init__(self):
        self.format = pyaudio.paInt16
        self.sample_rate = 44100
        self.channels = 2
        self.chunk_size = 1024

        self.p = pyaudio.PyAudio()  # Initialize PyAudio

        # Open a stream to capture audio from the USB microphone
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

    def record_loop_count(self, duration):
        return self.sample_rate / self.chunk_size * duration

    # def read(self):
    #     data = self.stream.read(self.chunk_size)
    #     # frames.append(data)
    #     return data

    def detect_sound(self, threshold) -> bool:
        input_data = self.stream.read(
            self.chunk_size
        )  # Read binary audio data from the stream
        input_array = np.frombuffer(
            input_data, dtype=np.int16
        )  # Convert binary data to a NumPy array
        above_threshold = (
            np.max(np.abs(input_array)) > threshold
        )  # exceeds the threshold?
        if above_threshold:
            return True
        else:
            return False

    def disconnect(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def save_record(self, output_file, frames):
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))
