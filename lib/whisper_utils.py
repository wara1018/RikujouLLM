from faster_whisper import WhisperModel

class WhisperModelWrapper:
    def __init__(self):
        self.model_size_or_path = "large-v3"
        self.model = WhisperModel(
            self.model_size_or_path, device = "cuda", compute_type = "float16"
        )
        
    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio=audio, beam_size=10, language="ja", without_timestamps=True
        )
        return segments