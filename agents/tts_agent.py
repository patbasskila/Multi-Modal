import torch
from TTS.api import TTS
import numpy as np

class TTSAgent:
    def __init__(self):
        # Only enable GPU if CUDA is actually available
        use_gpu = torch.cuda.is_available()
        if not use_gpu:
            print("CUDA not detected—falling back to CPU for TTS.")
        self.tts = TTS(
            "tts_models/en/ljspeech/tacotron2-DDC_ph",
            gpu=use_gpu
        )

    def run(self, text: str) -> dict:
        # Generate waveform (float32 NumPy array)
        wav = self.tts.tts(text)

        # Convert float32 array to 16-bit PCM bytes
        pcm = (wav * 32767).astype(np.int16).tobytes()
        return {"audio_bytes": pcm}
