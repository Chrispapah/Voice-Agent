"""Google Cloud Speech-to-Text streaming transcriber with API key auth and silence-based finals."""

from __future__ import annotations

import queue
from threading import Lock, Timer

from loguru import logger
from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import (
    GoogleTranscriberConfig,
    TranscriberType,
    Transcription,
)
from vocode.streaming.transcriber.base_transcriber import BaseThreadAsyncTranscriber

class SDRGoogleTranscriberConfig(GoogleTranscriberConfig, type=TranscriberType.GOOGLE.value):  # type: ignore
    """Extends vocode's config with API key + client-side utterance silence (ms after last interim)."""

    google_api_key: str = ""
    utterance_silence_ms: int = 300


class SDRGoogleTranscriber(BaseThreadAsyncTranscriber[SDRGoogleTranscriberConfig]):
    """Streaming STT via google-cloud-speech with API key; optional 300ms-style silence finals."""

    def __init__(self, transcriber_config: SDRGoogleTranscriberConfig):
        super().__init__(transcriber_config)

        from google.api_core.client_options import ClientOptions
        from google.cloud import speech

        self.speech = speech
        self._ended = False
        self.google_streaming_config = self.create_google_streaming_config()
        self.client = speech.SpeechClient(
            client_options=ClientOptions(api_key=transcriber_config.google_api_key)
        )

        self._silence_s = max(transcriber_config.utterance_silence_ms, 50) / 1000.0
        self._timer: Timer | None = None
        self._timer_lock = Lock()
        self._pending_interim_text: str | None = None
        self._pending_confidence: float = 0.0
        self._generation = 0
        self._last_timer_emitted_text: str | None = None

    def create_google_streaming_config(self):
        extra_params: dict = {}
        if self.transcriber_config.model:
            extra_params["model"] = self.transcriber_config.model
            extra_params["use_enhanced"] = True

        if self.transcriber_config.language_code:
            extra_params["language_code"] = self.transcriber_config.language_code

        if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            google_audio_encoding = self.speech.RecognitionConfig.AudioEncoding.LINEAR16
        elif self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            google_audio_encoding = self.speech.RecognitionConfig.AudioEncoding.MULAW
        else:
            raise ValueError(f"Unsupported audio encoding {self.transcriber_config.audio_encoding}")

        return self.speech.StreamingRecognitionConfig(
            config=self.speech.RecognitionConfig(
                encoding=google_audio_encoding,
                sample_rate_hertz=self.transcriber_config.sampling_rate,
                **extra_params,
            ),
            interim_results=True,
        )

    def _cancel_timer(self) -> None:
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _schedule_silence_final(self) -> None:
        with self._timer_lock:
            if self._timer is not None:
                self._timer.cancel()
            self._generation += 1
            gen = self._generation

            def fire() -> None:
                if gen != self._generation:
                    return
                text = self._pending_interim_text
                if not text or not text.strip():
                    return
                stripped = text.strip()
                self._last_timer_emitted_text = stripped
                self.output_janus_queue.sync_q.put_nowait(
                    Transcription(
                        message=text,
                        confidence=self._pending_confidence,
                        is_final=True,
                    )
                )

            self._timer = Timer(self._silence_s, fire)
            self._timer.daemon = True
            self._timer.start()

    def _run_loop(self):
        stream = self.generator()
        requests = (
            self.speech.StreamingRecognizeRequest(audio_content=content) for content in stream
        )
        url_hint = "speech.googleapis.com (API key)"
        logger.info("Google Speech streaming_recognize connecting via {}", url_hint)
        try:
            responses = self.client.streaming_recognize(self.google_streaming_config, requests)
            self.process_responses_loop(responses)
        except Exception:
            logger.exception("Google Speech transcriber loop crashed")
            raise

    def terminate(self):
        self._ended = True
        self._cancel_timer()
        super().terminate()

    def process_responses_loop(self, responses):
        for response in responses:
            self._on_response(response)
            if self._ended:
                break

    def _on_response(self, response):
        if not response.results:
            return

        result = response.results[0]
        if not result.alternatives:
            return

        top_choice = result.alternatives[0]
        message = top_choice.transcript
        confidence = top_choice.confidence or 0.0
        is_final = result.is_final

        if is_final:
            self._cancel_timer()
            if message.strip() and message.strip() == self._last_timer_emitted_text:
                self._last_timer_emitted_text = None
                return
            self._last_timer_emitted_text = None
            self.output_janus_queue.sync_q.put_nowait(
                Transcription(message=message, confidence=confidence, is_final=True)
            )
            return

        if not message.strip():
            return

        self.output_janus_queue.sync_q.put_nowait(
            Transcription(message=message, confidence=confidence, is_final=False)
        )
        self._pending_interim_text = message
        self._pending_confidence = confidence
        self._schedule_silence_final()

    def generator(self):
        while not self._ended:
            chunk = self.input_janus_queue.sync_q.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self.input_janus_queue.sync_q.get_nowait()
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            joined = b"".join(data)
            yield joined
