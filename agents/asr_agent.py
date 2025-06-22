import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

class ASRAgent:
    def __init__(self):
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")

        # Create the ASR pipeline, explicitly giving feature_extractor & tokenizer
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )


    def run(self, audio_bytes: bytes) -> dict:
        # Feed raw bytes to the pipeline (it will auto-resample / decode internally).
        result = self.pipe(audio_bytes)
        return {
            "text": result["text"],
            # Whisper pipelines put their confidence under "score" (if available),
            # otherwise default to 1.0
            "confidence": result.get("score", 1.0),
        }

