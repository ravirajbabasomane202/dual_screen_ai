"""Turn an exception (or an error message) into what a person needs: a plain-English cause, a recommended action and
the technical details. Used by the inline red banners and toasts; the full stack trace still goes to errors.log."""
from __future__ import annotations

import traceback
from dataclasses import dataclass


@dataclass
class ErrorInfo:
    cause: str
    action: str = ""
    details: str = ""
    kind: str = ""
    page: str = ""              # settings page that can fix it ("LLM", "Audio", ...)

    @property
    def message(self) -> str:
        return self.cause + (f"  {self.action}" if self.action else "")


_BY_TYPE = {
    "LLMNotRunning": ("The local language model server is not running.", "Start Ollama (run “ollama serve”) and try again.", "LLM"),
    "LLMModelMissing": ("The selected language model is not installed.", "Open Settings → LLM and press “Pull model”.", "LLM"),
    "LLMTimeout": ("The language model took too long to answer.", "Use a smaller model or a shorter answer length, or raise the timeout.", "LLM"),
    "LLMContextOverflow": ("The prompt is larger than the model's context window.", "Lower “Retrieved chunks” or raise the context window.", "LLM"),
    "LLMBadResponse": ("The language model returned an unusable (empty or malformed) response.", "Run Settings → Diagnostics, or check logs/llm.log.", "Diagnostics"),
    "AnswerRejected": ("The model kept describing the question instead of answering it.", "Try again, or choose a larger model.", "LLM"),
    "ResumeParseError": ("The resume could not be read. Your previous resume was kept.", "Try a PDF with selectable text, a DOCX or a TXT file.", "Resume"),
    "FileTooLarge": ("The file is too large.", "Use a smaller file.", "Resume"),
    "KnowledgeBaseError": ("The knowledge base could not be updated.", "Check the file, then try again.", "Knowledge Base"),
    "NoAudioDevice": ("No audio output device was found.", "Check your sound settings and the device selected in Settings → Audio.", "Audio"),
    "WasapiUnavailable": ("System-audio capture (WASAPI loopback) is not available.", "This feature needs Windows 10 or 11.", "Audio"),
    "AudioStreamFailure": ("The audio stream stopped unexpectedly.", "The app is reconnecting; check the selected device if it persists.", "Audio"),
    "WhisperModelMissing": ("The speech model is not downloaded.", "Open Settings → Speech and download a model.", "Speech"),
    "InsufficientMemory": ("There is not enough memory for this model.", "Choose a smaller speech or language model.", "Speech"),
    "ThreadAffinityError": ("A background task touched the user interface from the wrong thread.", "This is a bug; please export the diagnostics bundle.", "Debug & Logs"),
    "PermissionError": ("The file could not be accessed.", "Close any program using it and check the permissions.", ""),
    "FileNotFoundError": ("The file or folder no longer exists.", "Choose it again.", ""),
    "ConnectionError": ("A connection to a local service failed.", "Make sure the service is running.", "LLM"),
    "OSError": ("The operating system reported an error.", "Check the file path, disk space and permissions.", ""),
}
_BY_TEXT = (
    ("not running", "LLMNotRunning"), ("connection refused", "ConnectionError"), ("timed out", "LLMTimeout"),
    ("context length", "LLMContextOverflow"), ("no space left", "OSError"),
)


def describe(err, tb: bool = True) -> ErrorInfo:
    """`err` may be an exception or a string."""
    if isinstance(err, BaseException):
        kind = type(err).__name__
        text = getattr(err, "user_message", None) or str(err) or kind
        details = "".join(traceback.format_exception(type(err), err, err.__traceback__)) if tb else str(err)
    else:
        text, details = str(err), str(err)
        kind = next((k for needle, k in _BY_TEXT if needle in text.lower()), "")
    cause, action, page = _BY_TYPE.get(kind, ("", "", ""))
    if kind in _BY_TYPE and isinstance(err, BaseException) and getattr(err, "user_message", None):
        user = err.user_message.strip()
        if user and user != cause and kind in ("ResumeParseError", "FileTooLarge", "KnowledgeBaseError"):
            cause = user.splitlines()[0]                    # these carry the precise reason ("x.pdf: password-protected")
    if not cause:
        first = (text.strip().splitlines() or [kind or "Something went wrong."])[0]
        cause, action = first, "See the details below, or open the activity log."
    return ErrorInfo(cause, action, details, kind, page)
