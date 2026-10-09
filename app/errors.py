"""Friendly, user-facing errors. Every one carries a message safe to show in the UI."""


class AppError(Exception):
    user_message = "Something went wrong."

    def __init__(self, message: str | None = None):
        self.user_message = message or type(self).user_message
        super().__init__(self.user_message)


class NoAudioDevice(AppError):
    user_message = "No audio output device found.\n\nCheck Windows sound settings and try again."


class WasapiUnavailable(AppError):
    user_message = "WASAPI loopback is unavailable on this system.\n\nThis app requires Windows 10/11."


class AudioStreamFailure(AppError):
    user_message = "The audio stream stopped unexpectedly. Trying to reconnect..."


class WhisperModelMissing(AppError):
    user_message = "The speech model is not downloaded.\n\nOpen Setup and download a Whisper model."


class LLMNotRunning(AppError):
    user_message = "Ollama is not running.\n\nStart Ollama and try again."


class LLMModelMissing(AppError):
    user_message = "The selected language model is not installed.\n\nDownload it from Setup (or run: ollama pull <model>)."


class LLMTimeout(AppError):
    user_message = "The local language model took too long to respond.\n\nTry a smaller model or a shorter answer length."


class InsufficientMemory(AppError):
    user_message = "Not enough memory for this model.\n\nChoose a smaller Whisper/LLM model in Settings."


class NoSecondMonitor(AppError):
    user_message = "Only one monitor detected. The assistant window is on your only screen."


class LLMContextOverflow(AppError):
    user_message = ("The prompt is larger than the model's context window.\n\n"
                    "Lower 'Retrieved chunks' / 'Max tokens' in Settings, or raise the model's context window.")


class LLMBadResponse(AppError):
    user_message = ("The language model returned an unusable response (empty or malformed).\n\n"
                    "Open Settings → Debug → Run Diagnostics, or check logs/llm.log.")


class AnswerRejected(AppError):
    user_message = ("The model kept describing the question instead of answering it.\n\n"
                    "The answer was discarded. Try again, or use a larger model (see logs/llm.log).")


class ResumeParseError(AppError):
    user_message = "The resume could not be read. Your previous resume (if any) was kept."


class KnowledgeBaseError(AppError):
    user_message = "The knowledge base could not be updated."


class Cancelled(Exception):
    """A background operation was cancelled by the user (cooperative cancellation; never a failure)."""
    cancelled = True
    user_message = "Cancelled."

    def __init__(self, message: str = "Cancelled."):
        super().__init__(message)


class FileTooLarge(AppError):
    user_message = "The file is too large."
