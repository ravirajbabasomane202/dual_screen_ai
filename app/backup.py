"""Export / import everything the user created (settings, resume profiles, knowledge-base notes and Q&A, interview
target, detection feedback) as one zip, and the privacy helpers ("what is stored", "wipe all data")."""
from __future__ import annotations

import base64
import json
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

from app.errors import AppError
from settings.settings_manager import DEFAULTS

FORMAT = 1


class BackupError(AppError):
    user_message = "The backup file could not be read."


def export_backup(path, db, settings, services=None) -> dict:
    """Write the backup zip. Returns counts. Originals of resume files are included so profiles can be re-parsed."""
    from context.interview_profile import InterviewProfileStore
    from database.repositories import ResumeRepository
    counts = {"settings": 0, "resume_profiles": 0, "kb_entries": 0, "feedback": 0}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        kv = {k: v for k, v in db.all_kv().items() if not k.startswith("_")}
        counts["settings"] = len(kv)
        z.writestr("settings.json", json.dumps(kv, indent=1))
        repo = ResumeRepository(db)
        profiles = []
        for r in repo.list_profiles():
            full = repo.get(r.id, with_original=True)
            profiles.append({"label": r.label, "filename": r.filename, "active": r.active, "edits": r.edits,
                             "file": f"resume/{r.id}_{Path(r.filename).name}" if full.original else ""})
            if full.original:
                z.writestr(f"resume/{r.id}_{Path(r.filename).name}", full.original)
        counts["resume_profiles"] = len(profiles)
        z.writestr("resume/profiles.json", json.dumps(profiles, indent=1))
        entries = []
        if services is not None:
            entries = [{k: v for k, v in e.__dict__.items()} for e in services.kb.entries()]
        counts["kb_entries"] = len(entries)
        z.writestr("knowledge/entries.json", json.dumps(entries, indent=1))
        z.writestr("profile.json", json.dumps(asdict(InterviewProfileStore(db).load()), indent=1))
        fb = [{"text": r[0], "label": r[1], "detected": r[2], "confidence": r[3], "action": r[4]} for r in
              db.query("SELECT text,label,detected,confidence,action FROM question_feedback")]
        counts["feedback"] = len(fb)
        z.writestr("feedback.json", json.dumps(fb, indent=1))
        z.writestr("manifest.json", json.dumps({"app": "dual-screen-ai", "format": FORMAT, "created": time.time(),
                                                 "counts": counts}))
    return counts


def import_backup(path, db, settings, services=None) -> dict:
    """Merge a backup into the running app. Existing data is kept; duplicates are skipped."""
    from context.interview_profile import InterviewProfile, InterviewProfileStore
    from question.feedback import FeedbackStore
    done = {"settings": 0, "resume_profiles": 0, "kb_entries": 0, "feedback": 0, "skipped": 0}
    try:
        z = zipfile.ZipFile(path)
        manifest = json.loads(z.read("manifest.json"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as e:
        raise BackupError("This is not a backup made by this application.") from e
    if manifest.get("app") != "dual-screen-ai" or manifest.get("format", 0) > FORMAT:
        raise BackupError("This backup was made by a newer or different application.")
    with z:
        kv = json.loads(z.read("settings.json"))
        known = {f"{s}.{k}": v for s, d in DEFAULTS.items() for k, v in d.items()}
        apply = {k: v for k, v in kv.items() if k in known and k != "app.first_run_done"}
        settings.set_many(apply)
        done["settings"] = len(apply)
        if services is not None and services.resume is not None:
            existing = {(p.filename, p.file_hash) for p in services.resume.profiles()}
            import hashlib
            active_id = None
            for item in json.loads(z.read("resume/profiles.json")):
                if not item.get("file"):
                    done["skipped"] += 1
                    continue
                data = z.read(item["file"])
                if (item["filename"], hashlib.sha256(data).hexdigest()) in existing:
                    done["skipped"] += 1
                    continue
                res = services.resume.upload_bytes(item["filename"], data, new_profile=True, label=item.get("label", ""))
                if item.get("edits"):
                    services.resume.edit_fields(item["edits"], res.id)
                done["resume_profiles"] += 1
                if item.get("active"):
                    active_id = res.id
            if active_id is not None:
                services.resume.activate(active_id)
        if services is not None:
            have = {(e.kind, e.title, e.body, e.question, e.answer) for e in services.kb.entries()}
            for e in json.loads(z.read("knowledge/entries.json")):
                key = (e["kind"], e["title"], e["body"], e["question"], e["answer"])
                if key in have:
                    done["skipped"] += 1
                    continue
                if e["kind"] == "qa":
                    services.kb.add_qa(e["question"], e["answer"], e.get("tags", ""), e.get("title", ""))
                else:
                    services.kb.add_note(e["title"], e["body"], e.get("tags", ""))
                done["kb_entries"] += 1
        prof = json.loads(z.read("profile.json"))
        store = InterviewProfileStore(db)
        if store.load().empty and any(prof.values()):
            store.save(InterviewProfile(**{k: str(v) for k, v in prof.items() if k in InterviewProfile.__dataclass_fields__}))
        fb = FeedbackStore(db)
        for r in json.loads(z.read("feedback.json")):
            try:
                fb.record(r["text"], r["label"], r.get("detected", ""), r.get("confidence", 0.0), r.get("action", ""))
                done["feedback"] += 1
            except ValueError:
                done["skipped"] += 1
    return done


# ------------------------------------------------------------------------------------------ privacy
def stored_data_report(db, settings, log_dir=None, services=None) -> list[tuple[str, str, bool]]:
    """(what, how much / where, currently being collected)"""
    q = lambda sql: db.query_one(sql)[0]
    rows = [
        ("Question & answer history", f"{q('SELECT COUNT(*) FROM questions')} item(s)", bool(settings.get("privacy.store_questions"))),
        ("Conversation context", f"{q('SELECT COUNT(*) FROM context')} item(s), interview memory "
                                 f"{'saved' if q('SELECT COUNT(*) FROM interview_context') else 'empty'}", bool(settings.get("privacy.store_questions"))),
        ("Transcript file", "transcript.txt in the app folder", bool(settings.get("privacy.store_transcript"))),
        ("Audio recordings", "never written unless enabled", bool(settings.get("privacy.store_audio"))),
        ("Resume profiles", f"{q('SELECT COUNT(*) FROM resume_profiles')} profile(s), "
                            f"{q('SELECT COUNT(*) FROM resume_versions')} history version(s)", True),
        ("Knowledge base", f"{q('SELECT COUNT(*) FROM kb_documents')} document(s), {q('SELECT COUNT(*) FROM kb_chunks')} chunk(s)", True),
        ("Interview target", "company / role / job description" if q("SELECT COUNT(*) FROM interview_profile") else "empty", True),
        ("Feedback", f"{q('SELECT COUNT(*) FROM question_feedback')} detection correction(s), "
                     f"{q('SELECT COUNT(*) FROM answer_feedback')} answer rating(s), {q('SELECT COUNT(*) FROM pinned_answers')} pinned", True),
        ("Logs", f"{log_dir or 'logs folder'} (redacted unless debug mode is on)", True),
    ]
    return rows


def wipe_all_data(db, settings, log_dir=None, services=None, keep_settings: bool = True) -> dict:
    """Delete everything the app stored about the user. Settings survive unless keep_settings=False."""
    n = {}
    for table in ("questions", "context", "interview_context", "resume_versions", "resume_profiles", "kb_chunks",
                  "kb_documents", "kb_entries", "kb_meta", "interview_profile", "question_feedback", "answer_feedback",
                  "pinned_answers"):
        n[table] = db.query_one(f"SELECT COUNT(*) FROM {table}")[0]
    with db.transaction() as c:
        for table in n:
            c.execute(f"DELETE FROM {table}")
        if not keep_settings:
            c.execute("DELETE FROM settings")
    if services is not None:
        services.resume._fire()
        services.kb._invalidate()
        services.context.clear(keep_profile=False)
        services.context.target_text = ""
    if log_dir:
        for f in Path(log_dir).glob("*.log*"):
            if f.name not in ("crash.log",):
                try:
                    f.write_text("", encoding="utf-8") if f.suffix == ".log" else f.unlink()
                except OSError:
                    pass
    tr = Path(db.path).parent / "transcript.txt" if db.path != ":memory:" else None
    if tr and tr.exists():
        tr.unlink()
    return n
