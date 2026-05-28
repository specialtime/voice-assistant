from __future__ import annotations

import json
from pathlib import Path


DEFAULT_FILLER_WORDS = {"eh", "em", "mmm", "mm", "uh", "um", "hmm"}


class AzureSpeechService:
    def __init__(
        self,
        speech_key: str,
        speech_region: str,
        voice_name: str,
        voice_style: str | None = None,
        min_confidence: float = 0.55,
        filler_words: set[str] | None = None,
    ) -> None:
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.voice_name = voice_name
        self.voice_style = (voice_style or "").strip() or None
        self.min_confidence = min_confidence
        self.filler_words = filler_words or DEFAULT_FILLER_WORDS

    @staticmethod
    def _extract_confidence(recognition_result: object, speechsdk: object) -> float | None:
        try:
            payload = recognition_result.properties.get(  # type: ignore[attr-defined]
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult  # type: ignore[attr-defined]
            )
        except Exception:
            return None

        if not payload:
            return None

        try:
            data = json.loads(payload)
            confidence = data["NBest"][0].get("Confidence")
            return float(confidence) if confidence is not None else None
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def transcribe_wav(self, wav_file: Path) -> str:
        if not self.speech_key or not self.speech_region:
            raise RuntimeError("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")

        try:
            import azure.cognitiveservices.speech as speechsdk  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("azure-cognitiveservices-speech is not installed") from exc

        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        audio_config = speechsdk.audio.AudioConfig(filename=str(wav_file))
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = recognizer.recognize_once_async().get()
        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return ""

        text = (result.text or "").strip()
        if not text:
            return ""

        tokens = text.lower().split()
        if tokens and all(token in self.filler_words for token in tokens):
            return ""

        confidence = self._extract_confidence(result, speechsdk)
        if confidence is not None and confidence < self.min_confidence:
            return ""

        return text

    def _apply_voice_style(self, speech_config: object, speechsdk: object) -> None:
        if not self.voice_style:
            return

        try:
            property_id = speechsdk.PropertyId.SpeechServiceConnection_SynthStyle  # type: ignore[attr-defined]
        except AttributeError:
            property_id = "SpeechServiceConnection_SynthStyle"

        speech_config.set_property(property_id, self.voice_style)  # type: ignore[call-arg]

    def speak_text(self, text: str) -> None:
        if not self.speech_key or not self.speech_region:
            raise RuntimeError("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")

        try:
            import azure.cognitiveservices.speech as speechsdk  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("azure-cognitiveservices-speech is not installed") from exc

        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.speech_synthesis_voice_name = self.voice_name
        self._apply_voice_style(speech_config, speechsdk)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = synthesizer.speak_text_async(text).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = speechsdk.CancellationDetails.from_result(result)
            raise RuntimeError(f"Speech synthesis failed: {details.reason} - {details.error_details}")
