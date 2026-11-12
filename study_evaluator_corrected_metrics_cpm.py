"""Phase A/B/C study evaluator for the gaze keyboard.

Wraps the CURRENT live keyboard (gaze_keyboard_live_ui_matched.py) without
rebuilding it, so the study runs on exactly the keyboard you calibrated on and
type with: same window mode, same suggestion strip, same key geometry, same
filters, dwell logic and LLM suggestions. This file only adds target prompts,
trial progression and measurement.

Keep it beside gaze_keyboard_live_ui_matched.py, autocomplete_2.py and the
project modules (module1/, module2/, module3/, extra_trees_gaze_mapper.py).

It reuses the live keyboard's argument parser, so the live keyboard options and study options are accepted together. Use the same live settings as your normal run, so
use the SAME flags you type with, e.g.:

    python study_evaluator_corrected.py P01 S1 \\
        --calibration-csv calibration_keyboard_auto2.csv \\
        --model models/L2CSNet_mpiigaze.pkl --fullscreen \\
        --dwell-ms 800 --n-suggestions 3 --llm-model llama3.1:8b --llm-timeout 15

Before the countdown the evaluator waits for the language model to answer one
tiny request (cold-start of e.g. llama3.1:8b can take longer than --llm-timeout),
so suggestions work from the first word. Skip with --no-llm-warmup.

No-suggestion baseline with identical layout: add  --llm-url ""
Everything used is written to config.json in the session folder.
"""
import argparse
import csv
import json
import random
import threading
import time
import tkinter as tk
import urllib.request
from pathlib import Path

try:
    import gaze_keyboard_live_ui_matched as live
except ImportError as exc:                                     # pragma: no cover
    raise SystemExit(
        f"Could not import gaze_keyboard_live_ui_matched.py ({exc}).\n"
        "Keep this file beside it, autocomplete_2.py and the project modules.")
if not hasattr(live, "build_parser"):                          # pragma: no cover
    raise SystemExit("gaze_keyboard_live_ui_matched.py is out of date "
                     "(no build_parser). Copy over the latest version.")

SUG_PREFIX = live.SUG_PREFIX
DEFAULT_WORDS = ["HELLO", "WORLD", "GREAT", "WATER", "TIME"]
DEFAULT_SENTENCES = [
    "I AM A STUDENT",
    "GOOD MORNING EVERYONE",
    "I LOVE MY FAMILY",
]
LETTERS = list("QWERTYUIOPASDFGHJKLZXCVBNM")
TARGET_GREEN = "#16a34a"
KEY_GRAY = "#374151"

# Options of the OLD live keyboard that the current one no longer has. They are
# accepted (so old command lines still run) but ignored.
LEGACY_VALUE_FLAGS = [
    "--gap-tolerance", "--alpha", "--innovation-limit", "--max-trend-step",
    "--y-weight", "--fixation-frac", "--fixation-min-px", "--fix-exit-frames",
    "--fix-min-samples", "--fix-y-scale", "--snap-frac", "--hysteresis-frac",
    "--row-hysteresis-frac",
]
LEGACY_BOOL_FLAGS = ["--no-thread"]


# ───────────────────────────────────────────────────────── LLM warm-up
class LLMWarmup(threading.Thread):
    """Sends one tiny chat request so the model is loaded BEFORE the study starts.
    state: 'pending' -> 'ready' | 'failed'.  Uses the same endpoint/model as the
    live keyboard's Suggester, but a much longer timeout than --llm-timeout."""

    def __init__(self, url, model, timeout):
        super().__init__(daemon=True)
        self.url, self.model, self.timeout = url, model, float(timeout)
        self.state = "pending"
        self.error = ""
        self.t0 = time.perf_counter()
        self.start()

    def run(self):
        body = json.dumps({
            "model": self.model, "stream": False, "max_tokens": 1,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Reply with OK."}],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
            self.state = "ready"
        except Exception as exc:                       # server down, bad model, timeout
            self.error = str(exc)
            self.state = "failed"

    def elapsed(self):
        return time.perf_counter() - self.t0


# ───────────────────────────────────────────────────────── text helpers
def normalize_text(value):
    """Uppercase letters and single spaces only, for comparison."""
    return " ".join("".join(ch if ch.isalpha() else " " for ch in value.upper()).split())


def levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


# ───────────────────────────────────────────────────────── CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="Phase A key-target, Phase B word and Phase C sentence trials "
                    "on the live gaze keyboard. Accepts all live-keyboard options.",
        parents=[live.build_parser(add_help=False)])
    p.add_argument("participant_id")
    p.add_argument("session_id")

    g = p.add_argument_group("study")
    g.add_argument("--phase-a-trials", type=int, default=20)
    g.add_argument("--trial-timeout", type=float, default=8.0,
                   help="Phase A seconds per target.")
    g.add_argument("--word-timeout", type=float, default=30.0)
    g.add_argument("--sentence-timeout", type=float, default=120.0)
    g.add_argument("--words", nargs="+", default=DEFAULT_WORDS)
    g.add_argument("--sentences", nargs="+", default=DEFAULT_SENTENCES,
                   help="One or more sentence targets for Phase C (quote each).")
    g.add_argument("--seed", type=int, default=None,
                   help="Seed for the Phase A target order (random if omitted; "
                        "the value used is saved in config.json).")
    g.add_argument("--output-root", default="interaction_data")
    g.add_argument("--countdown", type=int, default=3,
                   help="Seconds of 'get ready' once tracking and the LLM are ready.")
    g.add_argument("--phase-pause", type=float, default=3.0,
                   help="Seconds the instruction for each phase stays on screen.")
    g.add_argument("--inter-trial-ms", type=int, default=600)
    g.add_argument("--no-llm-warmup", dest="llm_warmup", action="store_false",
                   help="Do not wait for the language model before starting.")
    g.add_argument("--llm-warmup-timeout", type=float, default=180.0,
                   help="Max seconds to wait for the model's first reply.")

    legacy = p.add_argument_group("legacy (accepted, ignored)")
    for flag in LEGACY_VALUE_FLAGS:
        legacy.add_argument(flag, default=None, help=argparse.SUPPRESS)
    for flag in LEGACY_BOOL_FLAGS:
        legacy.add_argument(flag, action="store_true", help=argparse.SUPPRESS)

    a = p.parse_args()
    ignored = [f for f in LEGACY_VALUE_FLAGS + LEGACY_BOOL_FLAGS
               if getattr(a, f.lstrip("-").replace("-", "_")) not in (None, False)]
    if ignored:
        print("[study] ignoring options the current live keyboard no longer has: "
              + " ".join(ignored))
    return a


# ───────────────────────────────────────────────────────── study runner
class StudyRunner:
    INTRO = {
        "A": "PHASE A — look at each GREEN key and hold your gaze to select it",
        "B": "PHASE B — type each word shown here, using gaze",
        "C": "PHASE C — type each sentence shown here, using gaze",
    }
    EVENT_FIELDS = [
        "participant_id", "session_id", "phase", "trial_idx", "intended",
        "selected", "correct", "elapsed_s", "dwell_ms", "threshold_ms",
        "confidence", "cursor_x", "cursor_y", "typed_after", "timestamp_seconds",
    ]
    A_FIELDS = [
        "participant_id", "session_id", "trial_idx", "target", "outcome",
        "n_selections", "n_false_selections", "first_selection",
        "first_attempt_correct", "trial_time_s",
    ]
    TEXT_FIELDS = [
        "participant_id", "session_id", "trial_idx", "{target}", "transcribed",
        "outcome", "trial_time_s", "n_selections", "n_backspaces",
        "edit_distance", "character_error_rate", "wpm", "cpm",
    ]

    def __init__(self, app, args, session_dir):
        self.app = app
        self.args = args
        self.session_dir = Path(session_dir)
        self.rng = random.Random(args.seed)

        self.words = [w.upper() for w in args.words]
        if not self.words or any(not w.isalpha() for w in self.words):
            raise ValueError("--words must contain alphabetic words only")
        self.sentences = [normalize_text(s) for s in args.sentences]
        if not self.sentences or any(not s for s in self.sentences):
            raise ValueError("--sentences must contain at least one sentence with letters")
        self.targets = self._make_targets()
        self.plan = {"A": self.targets, "B": self.words, "C": self.sentences}
        self.idx = {"A": 0, "B": 0, "C": 0}
        self.intro_done = set()
        self.results = {"A": [], "B": [], "C": []}
        self.counts = {ph: dict(selections=0, false=0) for ph in "ABC"}

        self.warmup = None
        if (getattr(args, "llm_warmup", True) and args.llm_url
                and args.n_suggestions > 0):
            self.warmup = LLMWarmup(args.llm_url, args.llm_model,
                                    args.llm_warmup_timeout)

        self.phase = None
        self.target = None
        self.trial_idx = 0
        self.active = False            # a trial is currently accepting selections
        self.done = False
        self.summary_written = False
        self.highlighted_key = None
        self._reset_trial()

        # CSV outputs ------------------------------------------------------
        self.events_file, self.events = self._open_csv("selection_events.csv", self.EVENT_FIELDS)
        self.a_file, self.a_csv = self._open_csv("phaseA_trials.csv", self.A_FIELDS)
        self.b_file, self.b_csv = self._open_csv(
            "phaseB_trials.csv", [f.replace("{target}", "target_word") for f in self.TEXT_FIELDS])
        self.c_file, self.c_csv = self._open_csv(
            "phaseC_trials.csv", [f.replace("{target}", "target_sentence") for f in self.TEXT_FIELDS])

        # Prompt: overlaid EXACTLY on the status bar (plus its empty padding), so
        # the keyboard geometry is identical to calibration and live typing.
        self.prompt = tk.Label(app.root, text="", bg="#111827", fg="#facc15",
                               font=("Arial", 18, "bold"))
        self.prompt.place(in_=app.status, x=0, y=-6, relwidth=1.0, relheight=1.0, height=14)
        self.prompt.lift()

        # Hooks into the unmodified live app -------------------------------
        self.original_set_fill = app._set_fill
        app._set_fill = self._set_fill_override
        self.original_select = app._select
        app._select = self._on_selection
        self.original_refresh = app._refresh_suggestions
        app._refresh_suggestions = self._refresh_suggestions
        self.original_apply = app._apply_suggestions
        app._apply_suggestions = self._apply_suggestions_guard
        self.original_close = app.close
        app.close = self.close
        app.root.protocol("WM_DELETE_WINDOW", self.close)

        self._set_prompt("Looking for your face — please face the camera…")
        app.root.after(100, self._tick)
        app.root.after(200, self._wait_ready)

    # ── setup helpers ──────────────────────────────────────────────────────
    def _open_csv(self, name, fields):
        f = open(self.session_dir / name, "w", newline="", encoding="utf-8-sig")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        return f, w

    def _make_targets(self):
        targets, previous = [], None
        for _ in range(self.args.phase_a_trials):
            target = self.rng.choice([k for k in LETTERS if k != previous])
            targets.append(target)
            previous = target
        return targets

    def _reset_trial(self):
        self.trial_start = time.perf_counter()
        self.first_selection = None
        self.trial_selections = 0
        self.false_selections = 0
        self.backspaces = 0
        self.selection_started = False  # first selection of the trial has begun

    # ── small UI helpers ───────────────────────────────────────────────────
    def _set_prompt(self, text):
        self.prompt.config(text=text)

    def _clear_output(self):
        self.app.output.delete("1.0", "end")

    def _blank_suggestions(self):
        app = self.app
        if getattr(app, "sug_keys", None) and any(app.sug_cur):
            app._apply_suggestions([None] * len(app.sug_keys))

    def _refresh_suggestions(self):
        """Suggestions only run during Phase B/C trials; Phase A stays quiet so the
        LLM does not steal CPU from tracking and no stale words are shown."""
        if self.active and self.phase in ("B", "C"):
            self.original_refresh()
        else:
            self._blank_suggestions()

    def _apply_suggestions_guard(self, lst, hint=""):
        """The live loop applies late LLM replies whenever they arrive. Outside a
        B/C trial, or before the first selection of a trial (empty text, exactly as
        in the live keyboard), a late reply belongs to the PREVIOUS word: drop it."""
        lst = list(lst)
        allowed = (self.active and self.phase in ("B", "C")
                   and self.selection_started)
        if not allowed and any(lst):
            lst = [None] * len(lst)
        return self.original_apply(lst, hint)

    # ── Phase A target highlight (steady green, no flicker) ────────────────
    def _set_fill_override(self, key, color):
        if self.active and self.phase == "A" and key == self.target:
            color = TARGET_GREEN
        self.original_set_fill(key, color)

    def _show_target_key_green(self):
        new_key = self.target if (self.active and self.phase == "A") else None
        if self.highlighted_key and self.highlighted_key != new_key:
            self.original_set_fill(self.highlighted_key, KEY_GRAY)
        self.highlighted_key = new_key
        if new_key:
            self.original_set_fill(new_key, TARGET_GREEN)

    def _clear_highlight(self):
        if self.highlighted_key:
            self.original_set_fill(self.highlighted_key, KEY_GRAY)
            self.highlighted_key = None

    def _ensure_highlight(self):
        # The live canvas is rebuilt on resize (e.g. while fullscreen settles),
        # which drops the highlight; re-assert it.
        if self.target and self.app.key_fill.get(self.target) != TARGET_GREEN:
            self._show_target_key_green()

    # ── start-up gate and phase flow ───────────────────────────────────────
    def _wait_ready(self):
        if self.done:
            return
        app = self.app
        tracking_ok = getattr(app, "cx", None) is not None
        if not tracking_ok:
            self._set_prompt("Looking for your face — please face the camera…")
            app.root.after(200, self._wait_ready)
            return
        w = self.warmup
        if w is not None and w.state == "pending":
            self._set_prompt(f"Loading the language model…  {w.elapsed():.0f}s")
            app.root.after(200, self._wait_ready)
            return
        if w is not None and w.state == "failed":
            print(f"[study] WARNING: LLM warm-up failed ({w.error}). "
                  "Suggestions may be empty. Check --llm-url / --llm-model.")
            self._set_prompt("Language model not responding — suggestions may be empty")
            self.warmup = None
            app.root.after(2500, self._wait_ready)
            return
        self._countdown(max(0, int(self.args.countdown)))

    def _countdown(self, n):
        if self.done:
            return
        if n <= 0:
            self._advance()
            return
        self._set_prompt(f"Get ready…  {n}")
        self.app.root.after(1000, lambda: self._countdown(n - 1))

    def _advance(self):
        if self.done:
            return
        for ph in ("A", "B", "C"):
            if self.idx[ph] < len(self.plan[ph]):
                self._begin(ph)
                return
        self._finish_study()

    def _begin(self, ph):
        if self.done:
            return
        self.phase = ph
        if self.idx[ph] == 0 and ph not in self.intro_done:      # phase instruction
            self.intro_done.add(ph)
            self.active = False
            self._clear_highlight()
            self._clear_output()
            self._blank_suggestions()
            self._set_prompt(self.INTRO[ph])
            self.app.root.after(int(self.args.phase_pause * 1000), lambda: self._begin(ph))
            return
        n = self.idx[ph]
        self.target = self.plan[ph][n]
        self.trial_idx = n + 1
        self._clear_output()
        self._blank_suggestions()
        self._reset_trial()
        self.active = True
        total = len(self.plan[ph])
        if ph == "A":
            self._show_target_key_green()
            self._set_prompt(f"PHASE A · TARGET {self.trial_idx}/{total}:  {self.target}")
        elif ph == "B":
            self._set_prompt(f"PHASE B · WORD {self.trial_idx}/{total}:  {self.target}")
        else:
            self._set_prompt(f"PHASE C · SENTENCE {self.trial_idx}/{total}:  {self.target}")

    def _limit(self):
        return {"A": self.args.trial_timeout, "B": self.args.word_timeout,
                "C": self.args.sentence_timeout}[self.phase]

    def _tick(self):
        if self.done:
            return
        try:
            if self.active:
                if self.phase == "A":
                    self._ensure_highlight()
                if time.perf_counter() - self.trial_start >= self._limit():
                    self._finish_trial("timeout")
        except Exception as exc:                                  # never kill the tick
            print(f"[study] tick error: {exc}")
        if not self.done:
            self.app.root.after(100, self._tick)

    # ── selection handling ─────────────────────────────────────────────────
    def _on_track(self, typed):
        """True if `typed` is a correct prefix of (or equal to) the target."""
        t = " ".join(typed.upper().split())
        if t == self.target:
            return True
        if typed.endswith(" ") and t:
            t += " "
        return self.target.startswith(t)

    def _on_selection(self, k, conf, sx, sy, cx, cy, dwell_ms, threshold):
        if self.done:
            return
        if not self.active:                       # between trials / instructions
            return
        app = self.app
        typed_before = app.output.get("1.0", "end-1c")
        self.selection_started = True

        # Run the ORIGINAL live selection (typing, drift learning, flash feedback).
        self.original_select(k, conf, sx, sy, cx, cy, dwell_ms, threshold)
        typed_after = app.output.get("1.0", "end-1c")
        if self.phase == "A" and typed_after != typed_before:
            # Phase A is target selection, not typing: keep the text box unchanged.
            app.output.delete("1.0", "end")
            app.output.insert("1.0", typed_before)
            typed_after = typed_before

        now = time.perf_counter()
        ph = self.phase
        c = self.counts[ph]
        c["selections"] += 1
        self.trial_selections += 1
        if self.first_selection is None:
            self.first_selection = k

        false_sel = 0
        correct = ""
        if ph == "A":
            correct = int(k == self.target)
            false_sel = int(not correct)
        else:
            if k == "⌫":
                self.backspaces += 1
            elif k not in ("SPACE", "ENTER"):
                target_norm = normalize_text(self.target)
                typed_norm = normalize_text(typed_after)
                if not target_norm.startswith(typed_norm):
                    false_sel = 1
        if false_sel:
            self.false_selections += 1
            c["false"] += 1

        self.events.writerow({
            "participant_id": self.args.participant_id,
            "session_id": self.args.session_id, "phase": ph,
            "trial_idx": self.trial_idx, "intended": self.target,
            "selected": k, "correct": correct,
            "elapsed_s": round(now - self.trial_start, 4),
            "dwell_ms": round(float(dwell_ms), 2),
            "threshold_ms": round(float(threshold), 2),
            "confidence": round(float(conf), 5),
            "cursor_x": round(float(cx), 2), "cursor_y": round(float(cy), 2),
            "typed_after": typed_after, "timestamp_seconds": round(now, 6),
        })
        self.events_file.flush()

        if ph == "A":
            if k == self.target:
                self._finish_trial("success")
        elif normalize_text(typed_after) == self.target:
            self._finish_trial("completed")

    # ── trial completion ───────────────────────────────────────────────────
    def _finish_trial(self, outcome):
        if not self.active or self.done:               # guards success/timeout races
            return
        self.active = False
        ph = self.phase
        elapsed = time.perf_counter() - self.trial_start
        base = {"participant_id": self.args.participant_id,
                "session_id": self.args.session_id, "trial_idx": self.trial_idx}
        if ph == "A":
            first_correct = int(self.first_selection == self.target)
            self.a_csv.writerow({
                **base, "target": self.target, "outcome": outcome,
                "n_selections": self.trial_selections,
                "n_false_selections": self.false_selections,
                "first_selection": self.first_selection or "",
                "first_attempt_correct": first_correct,
                "trial_time_s": round(elapsed, 4)})
            self.a_file.flush()
            self.results["A"].append({
                "outcome": outcome, "first_attempt_correct": first_correct,
                "n_selections": self.trial_selections,
                "trial_time_s": elapsed, "target": self.target})
            self._clear_highlight()
        else:
            typed = normalize_text(self.app.output.get("1.0", "end-1c"))
            target = self.target
            distance = levenshtein(target, typed)
            cer = distance / max(1, len(target))
            # WPM: characters typed, divided into 5-character words per minute.
            wpm = (len(typed) / 5.0) / max(elapsed / 60.0, 1e-9)
            cpm = len(typed) / max(elapsed / 60.0, 1e-9)
            key = "target_word" if ph == "B" else "target_sentence"
            row = {
                **base, key: target, "transcribed": typed, "outcome": outcome,
                "trial_time_s": round(elapsed, 4),
                "n_selections": self.trial_selections,
                "n_backspaces": self.backspaces,
                "edit_distance": distance,
                "character_error_rate": round(cer, 5), "wpm": round(wpm, 4),
                "cpm": round(cpm, 4)}
            writer, fh = (self.b_csv, self.b_file) if ph == "B" else (self.c_csv, self.c_file)
            writer.writerow(row)
            fh.flush()
            self.results[ph].append({
                "outcome": outcome, "trial_time_s": elapsed,
                "n_selections": self.trial_selections,
                "edit_distance": distance, "character_error_rate": cer,
                "wpm": wpm, "cpm": cpm, "target": target, "transcribed": typed})
        self.idx[ph] += 1
        self.app.root.after(self.args.inter_trial_ms, self._advance)

    # ── summaries ──────────────────────────────────────────────────────────
    @staticmethod
    def _mean(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    @staticmethod
    def _median(values):
        if not values:
            return 0.0
        v = sorted(values)
        mid = len(v) // 2
        return round(v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2, 4)

    def _phase_summary(self, ph, kind):
        rows = self.results[ph]
        requested = len(self.plan[ph])
        false_count = self.counts[ph]["false"]
        selection_count = self.counts[ph]["selections"]
        active = sum(float(r.get("trial_time_s", 0)) for r in rows)
        common = {
            "false_selections": false_count,
            "false_selections_per_minute": round(false_count / (active / 60), 4) if active else 0.0,
            "false_per_selection": round(false_count / selection_count, 4) if selection_count else 0.0,
            "active_time_s": round(active, 3),
        }
        if ph == "A":
            successes = sum(r.get("outcome") == "success" for r in rows)
            first_correct = sum(int(r.get("first_attempt_correct", 0)) for r in rows)
            successful_times = [
                float(r["trial_time_s"]) for r in rows if r.get("outcome") == "success"
            ]
            return {
                "trials": requested,
                "completed_trials": min(len(rows), requested),
                "successes": successes,
                "success_rate": round(successes / requested, 4) if requested else 0.0,
                "first_attempt_accuracy": round(first_correct / len(rows), 4) if rows else 0.0,
                **common,
                "mean_time_to_select_s": self._mean(successful_times),
                "median_time_to_select_s": self._median(successful_times),
            }
        completed = sum(r.get("outcome") == "completed" for r in rows)
        target_chars = sum(len(r.get("target", "")) for r in rows)
        total_selections = sum(int(r.get("n_selections", 0)) for r in rows)
        return {
            kind: requested,
            "completed": completed,
            "mean_wpm": self._mean([float(r.get("wpm", 0)) for r in rows]),
            "mean_cpm": self._mean([float(r.get("cpm", 0)) for r in rows]),
            "mean_char_error_rate": self._mean(
                [float(r.get("character_error_rate", 0)) for r in rows]),
            "mean_kspc": round(total_selections / target_chars, 4) if target_chars else 0.0,
            **common,
        }

    def _build_summary(self, aborted):
        a = self.args
        return {
            "participant_id": a.participant_id,
            "session_id": a.session_id,
            "mode": a.mode,
            "phaseA": self._phase_summary("A", "trials"),
            "phaseB": self._phase_summary("B", "words"),
            "phaseC": self._phase_summary("C", "sentences"),
            "calibration_csv": str(Path(a.calibration).resolve()),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _write_summary(self, aborted):
        summary = self._build_summary(aborted)
        path = self.session_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.summary_written = True
        return summary, path

    def _finish_study(self):
        if self.done:
            return
        summary, path = self._write_summary(aborted=False)
        self.done = True
        self.active = False
        for worker in (getattr(self.app, "tracker", None), getattr(self.app, "grabber", None)):
            if worker is not None:
                worker.running = False            # stop camera/LLM load; results are in
        self._set_prompt("STUDY COMPLETE — thank you!   (ESC to close)")
        self.app.canvas.pack_forget()
        self.app.output.config(height=22, font=("Courier", 11), wrap="word")
        self.app.output.delete("1.0", "end")
        self.app.output.insert("1.0", json.dumps(summary, indent=2))
        self.app.output.pack(fill="both", expand=True, padx=12, pady=8)
        self.close_files()

    def close_files(self):
        for f in (self.events_file, self.a_file, self.b_file, self.c_file):
            try:
                if f and not f.closed:
                    f.flush()
                    f.close()
            except Exception:
                pass

    def close(self):
        if not self.summary_written:
            try:
                self._write_summary(aborted=True)     # keep partial results on ESC
            except Exception as exc:
                print(f"[study] could not write summary: {exc}")
        self.done = True
        self.close_files()
        self.original_close()


# ───────────────────────────────────────────────────────── main
def main():
    args = parse_args()
    if args.phase_a_trials < 1:
        raise ValueError("--phase-a-trials must be at least 1")
    calibration = Path(args.calibration).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(f"Calibration CSV not found: {calibration}")
    if not model.is_file():
        raise FileNotFoundError(f"Model file not found: {model}")
    args.calibration, args.model = str(calibration), str(model)
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(2 ** 31)

    session_dir = Path(args.output_root).expanduser().resolve() / args.participant_id / args.session_id
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(f"Session folder is not empty: {session_dir}. Use a new session ID.")
    session_dir.mkdir(parents=True, exist_ok=True)
    args.output = str(session_dir / "live_dwell_events.csv")      # live keyboard's own log

    config = dict(vars(args))
    config["live_script"] = str(Path(live.__file__).resolve())
    config["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (session_dir / "config.json").write_text(json.dumps(config, indent=2, default=str),
                                             encoding="utf-8")

    app = live.App(args)           # the unmodified live keyboard, same flags/defaults
    StudyRunner(app, args, session_dir)
    app.run()


if __name__ == "__main__":
    main()