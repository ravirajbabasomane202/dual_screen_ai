"""python -m diagnostics [--test-llm] [--debug]   (uses the real app database and settings)"""
import argparse
import sys

from app.paths import app_dir
from applog import configure
from database.sqlite import Database
from settings.settings_manager import SettingsManager


def main() -> int:
    ap = argparse.ArgumentParser(prog="diagnostics")
    ap.add_argument("--test-llm", action="store_true", help="only run the LLM test")
    ap.add_argument("--debug", action="store_true", help="verbose logs (also logs question/answer text)")
    ap.add_argument("--no-hardware", action="store_true", help="skip microphone / VAD / STT checks")
    a = ap.parse_args()
    configure(app_dir() / "logs", debug=a.debug)
    db = Database()
    settings = SettingsManager(db)
    if a.test_llm:
        from hardware.detector import detect, recommend
        from llm.diagnostics import test_llm
        from llm.model_manager import create_provider
        model = settings.get("ai.model")
        if model == "auto":
            model = recommend(detect()).llm_model
        r = test_llm(create_provider(settings, model), settings)
        print(r.format())
        return 0 if r.ok else 1
    from diagnostics.runner import run_full_diagnostics
    rep = run_full_diagnostics(settings, db, skip_hardware=a.no_hardware)
    print(rep.format())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
