import argparse
import csv
import math
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from module1.eye_processing import get_face
from module2.gaze_estimation import GazeEstimator
from module3.head_pose import estimate_head_pose
from extra_trees_gaze_mapper import ExtraTreesGazeMapper
from autocomplete_llm import Suggester, SUG_PREFIX   # [AC]

REQ = ('raw_gaze_yaw', 'raw_gaze_pitch', 'head_yaw', 'head_pitch',
       'target_x', 'target_y')

ROWS = [
    list('QWERTYUIOP'),
    list('ASDFGHJKL'),
    list('ZXCVBNM') + ['⌫', 'SPACE', 'ENTER'],
]

MAX_DWELL_DT_MS = 100.0      # clamp per-step dwell increment
UNDO_WINDOW_S = 6.0          # backspace within this time undoes last bias update


def apply_window_mode(root, fullscreen):
    """Identical helper lives in the calibration script - keep the two in sync.
    Fullscreen, or a maximized window, so both scripts get the same canvas size."""
    if fullscreen:
        root.attributes('-fullscreen', True)
        return
    try:
        root.state('zoomed')                      # Windows / macOS
    except tk.TclError:
        try:
            root.attributes('-zoomed', True)      # most Linux window managers
        except tk.TclError:
            root.geometry(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0')


# ─────────────────────────────────────────── filters
class OneEuro:
    """
    One Euro filter (Casiez et al. 2012), one axis.

    min_cutoff (Hz): smoothing when the signal is still.  LOWER = steadier
                     on a fixated key, slightly slower to start moving.
    beta:            how fast the cutoff rises with speed.  HIGHER = less lag
                     during fast moves, but a little more jitter.
    Speeds here are in px/s.
    """

    def __init__(self, min_cutoff=1.0, beta=0.004, d_cutoff=1.5):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self.t = None
        self.x = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x, t):
        x = float(x)
        if self.t is None:
            self.t, self.x, self.dx = t, x, 0.0
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x) / dt
        self.dx = a_d * dx + (1.0 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1.0 - a) * self.x
        return self.x


class DeadZone:
    """Cursor stabiliser. The output only moves once the input leaves a circle of
    `radius` px around it (then it follows, trailing by `radius`); inside the circle
    it eases toward the input by `pull` per frame. Kills small jitter on a fixated key.
    radius <= 0 disables it."""

    def __init__(self, radius=20.0, pull=0.15):
        self.r = float(radius)
        self.pull = float(pull)
        self.reset()

    def reset(self):
        self.x = self.y = None

    def update(self, x, y):
        if self.r <= 0:
            return x, y
        if self.x is None:
            self.x, self.y = x, y
            return x, y
        dx, dy = x - self.x, y - self.y
        d = math.hypot(dx, dy)
        if d > self.r:
            k = (d - self.r) / d
            self.x += dx * k
            self.y += dy * k
        else:
            self.x += dx * self.pull
            self.y += dy * self.pull
        return self.x, self.y


class MedianStabiliser:
    """Rolling per-axis median over the last n RAW samples (spike rejection)."""

    def __init__(self, n=3):
        self.n = max(1, int(n))
        self._x = deque(maxlen=self.n)
        self._y = deque(maxlen=self.n)

    def reset(self):
        self._x.clear()
        self._y.clear()

    def update(self, x, y):
        self._x.append(float(x))
        self._y.append(float(y))
        return float(np.median(self._x)), float(np.median(self._y))


# ─────────────────────────────────────────── calibration loader
def load_calibration(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    aliases = {
        'raw_gaze_yaw':   'raw_gaze_yaw_deg',
        'raw_gaze_pitch': 'raw_gaze_pitch_deg',
        'head_yaw':       'head_yaw_deg',
        'head_pitch':     'head_pitch_deg',
    }

    good = []
    for r in rows:
        try:
            vals = {}
            for k in REQ:
                src = k if r.get(k) not in (None, '') else aliases.get(k, k)
                vals[k] = float(r[src])
            if 'sample_valid' in r and str(r['sample_valid']).strip() not in (
                    '1', '1.0', 'True', 'true'):
                continue
            if all(math.isfinite(v) for v in vals.values()):
                good.append(vals)
        except (KeyError, ValueError, TypeError):
            continue

    if len(good) < 9:
        raise ValueError(f'Need ≥9 valid calibration rows; found {len(good)}')

    m = ExtraTreesGazeMapper()
    m.fit(
        gaze_yaw   = np.array([r['raw_gaze_yaw']  for r in good]),
        gaze_pitch = np.array([r['raw_gaze_pitch'] for r in good]),
        head_yaw   = np.array([r['head_yaw']       for r in good]),
        head_pitch = np.array([r['head_pitch']      for r in good]),
        target_x   = np.array([r['target_x']       for r in good]),
        target_y   = np.array([r['target_y']       for r in good]),
    )
    return m, len(good)


# ─────────────────────────────────────────── threading helpers
class FrameGrabber(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap     = cap
        self.lock    = threading.Lock()
        self.frame   = None
        self.seq     = 0
        self.running = True
        self.start()

    def run(self):
        while self.running:
            ret, f = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = f
                self.seq  += 1

    def latest(self, last_seq):
        with self.lock:
            if self.frame is None or self.seq == last_seq:
                return None, last_seq
            return self.frame.copy(), self.seq


class Tracker(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app        = app
        self.lock       = threading.Lock()
        self.result     = None
        self.rid        = 0
        self.last_taken = 0
        self.running    = True
        self.start()

    def run(self):
        seq = 0
        while self.running:
            frame, s2 = self.app.grabber.latest(seq)
            if frame is None:
                time.sleep(0.002)
                continue
            seq = s2
            try:
                r, err = self.app.process(frame), ''
            except Exception as e:
                r, err = None, str(e)
            t = time.perf_counter()          # stamp when the result was ready
            with self.lock:
                self.result = (r, err, t)
                self.rid   += 1

    def take(self):
        with self.lock:
            if self.rid == self.last_taken:
                return None
            self.last_taken = self.rid
            return self.result


# ─────────────────────────────────────────── main app
class App:
    def __init__(self, a):
        self.a = a

        self.mapper, n = load_calibration(a.calibration)
        print(f'Mapper fitted on {n} calibration samples')

        with open(a.calibration, newline='', encoding='utf-8-sig') as f:
            fr = next(csv.DictReader(f), {})
        self.cal_w = float(fr.get('screen_width')  or fr.get('screen_width_px')  or 1536)
        self.cal_h = float(fr.get('screen_height') or fr.get('screen_height_px') or 864)

        self.est = GazeEstimator(a.model, device='cuda' if a.cuda else 'cpu')
        self.cap = cv2.VideoCapture(a.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera {a.camera}')
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Filters: median on raw -> One Euro per axis
        self.median = MedianStabiliser(n=a.median_n)
        self.fx = OneEuro(a.min_cutoff, a.beta)
        self.fy = OneEuro(a.min_cutoff, a.beta)
        self.dz = DeadZone(a.deadzone_px)    # final cursor stabiliser
        # Input-angle smoothing (runs in the tracker thread, BEFORE the tree mapper,
        # whose staircase output otherwise flips between leaves on tiny angle noise)
        self.af = [OneEuro(a.angle_min_cutoff, a.angle_beta) for _ in range(4)]
        self.af_last_t = None
        self.res_times = deque(maxlen=20)    # tracker result timestamps -> fps readout
        self.pos_hist = deque(maxlen=4)      # (t, x, y) of filtered pos
        self.cx = None                       # last good cursor position
        self.cy = None

        # Drift-correction bias (canvas px), applied after filtering
        self.bias = [0.0, 0.0]
        self.last_bias_delta = None
        self.last_bias_t = 0.0

        # Tkinter UI
        self.root = tk.Tk()
        self.root.title('Gaze Virtual Keyboard')
        self.root.configure(bg='#111827')
        apply_window_mode(self.root, a.fullscreen)
        self.root.bind('<Escape>', lambda _e: self.close())
        self.root.bind('<F5>', lambda _e: self._reset_bias())

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        self.output = tk.Text(self.root, height=3, font=('Arial', 22), takefocus=0,
                              bg='#f3f4f6', fg='#111827')
        self.output.pack(fill='x', padx=12, pady=8)

        self.status = tk.Label(
            self.root, text='Starting…', bg='#111827', fg='white', font=('Arial', 12))
        self.status.pack(fill='x')

        # Suggestion slots are part of the canvas UI, matching the reference layout.
        n_sug = max(0, a.n_suggestions)
        self.sug_keys = [f'{SUG_PREFIX}{i + 1}' for i in range(n_sug)]
        self.rows = ROWS
        self.sug_label = None

        self.canvas = tk.Canvas(self.root, bg='#111827', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=10, pady=10)

        # Layout state
        self.rects       = {}
        self.centers     = {}
        self.row_of      = {}
        self.key_h       = 100.0
        self.layout_size = None
        self.key_items   = {}
        self.key_text_items = {}
        self.key_fill    = {}
        self.cursor_item = None
        self.bar_item    = None
        self.ox = self.oy = 0

        # [AC] autocomplete state
        self.sug = Suggester(n=n_sug, llm_url=a.llm_url,
                             llm_model=a.llm_model, llm_timeout=a.llm_timeout,
                             hint=a.llm_hint)
        self.sug_cur = [None] * n_sug
        self.sug_items = {}

        # Dwell state
        self.active          = None   # key currently being dwelled on
        self.candidate_key   = None   # key gaze is pointing at (unconfirmed)
        self.candidate_since = 0.0
        self.out_since       = None   # when cursor left active key (hysteresis)
        self.dwell_accum_ms  = 0.0
        self.dwell_pts       = deque(maxlen=15)
        self.last_dwell_t    = None
        self.repeating       = False
        self.last_select_t   = 0.0
        self.cascade_n       = 0      # consecutive fast selections (cascading dwell)
        self._n_sel          = 0      # selections so far (prefetch de-duplication)
        self._pf_for         = None   # (key, _n_sel) already prefetched
        self.flash_key       = None
        self.flash_until     = 0.0
        self.frame_count     = 0
        self.lost_since      = None

        # Logging
        self.path = Path(a.output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.path, 'w', newline='', encoding='utf-8')
        fields = [
            'timestamp_s', 'frame', 'mode', 'threshold_ms',
            'gaze_key', 'active_key', 'selected_key',
            'conf', 'sx', 'sy', 'cx', 'cy',
            'dwell_ms', 'stability', 'event',
        ]
        self.writer = csv.DictWriter(self.log, fieldnames=fields)
        self.writer.writeheader()

        self.grabber = FrameGrabber(self.cap)
        self.tracker = Tracker(self)

        self.root.after(50, self.loop)

    # ── logging ────────────────────────────────────────────────────────────
    def log_row(self, event='frame', gaze_key=None, selected=None,
                conf=None, sx=None, sy=None, cx=None, cy=None,
                dwell=None, stability=None, threshold=None):
        self.writer.writerow({
            'timestamp_s':  f'{time.perf_counter():.6f}',
            'frame':        self.frame_count,
            'mode':         self.a.mode,
            'threshold_ms': threshold or '',
            'gaze_key':     gaze_key  or '',
            'active_key':   self.active or '',
            'selected_key': selected  or '',
            'conf':    '' if conf      is None else round(conf, 4),
            'sx':      '' if sx        is None else round(sx, 1),
            'sy':      '' if sy        is None else round(sy, 1),
            'cx':      '' if cx        is None else round(cx, 1),
            'cy':      '' if cy        is None else round(cy, 1),
            'dwell_ms':     '' if dwell     is None else round(dwell, 1),
            'stability':    '' if stability is None else round(stability, 3),
            'event':        event,
        })
        self.log.flush()

    # ── gaze pipeline ──────────────────────────────────────────────────────
    def process(self, frame):
        frame = cv2.flip(frame, 1)
        hp = estimate_head_pose(frame)
        _, face = get_face(frame, self.a.face_size)
        if hp is None or face is None:
            raise ValueError('face/pose missing')
        yaw, pitch, conf = self.est.predict_gaze(face)
        for v in (yaw, pitch, conf, hp.get('yaw'), hp.get('pitch')):
            if v is None or not math.isfinite(float(v)):
                raise ValueError('invalid gaze values')
        gy, gp, hy, hpi = float(yaw), float(pitch), float(hp['yaw']), float(hp['pitch'])
        if self.a.angle_smooth:
            t = time.perf_counter()
            if self.af_last_t is not None and t - self.af_last_t > 0.5:
                for f in self.af:                 # face was lost: don't smooth across the gap
                    f.reset()
            self.af_last_t = t
            gy, gp, hy, hpi = (f.update(v, t) for f, v in
                               zip(self.af, (gy, gp, hy, hpi)))
        px, py = self.mapper.predict(
            [gy], [gp], head_yaw=[hy], head_pitch=[hpi],
        )
        rx = float(np.asarray(px).ravel()[0])
        ry = float(np.asarray(py).ravel()[0])
        rx = max(0.0, min(self.cal_w - 1, rx))
        ry = max(0.0, min(self.cal_h - 1, ry))
        return rx, ry, float(conf)

    # ── layout / canvas ────────────────────────────────────────────────────
    def _layout(self, cw, ch):
        gap = 8
        self.rects.clear()
        self.centers.clear()
        self.row_of.clear()

        # Dedicated suggestion strip at the top (only when suggestions are enabled),
        # followed by three full keyboard rows. Same formulas as the calibration script.
        self.suggestion_h = (max(52.0, min(82.0, ch * 0.12))
                             if self.sug_keys else 0.0)
        sug_gap = 8.0
        if self.sug_keys:
            sug_w = (cw - 16 - (len(self.sug_keys) - 1) * sug_gap) / len(self.sug_keys)
            for i, k in enumerate(self.sug_keys):
                x1 = 8 + i * (sug_w + sug_gap)
                x2 = x1 + sug_w
                y1, y2 = 4.0, self.suggestion_h - 4.0
                self.rects[k] = (x1, y1, x2, y2)
                self.centers[k] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                self.row_of[k] = -1

        keyboard_top = self.suggestion_h + 4.0 if self.sug_keys else 8.0
        keyboard_h = max(1.0, ch - keyboard_top - 8.0)
        rh = keyboard_h / len(ROWS)
        kh = rh - gap
        self.key_h = kh
        for ri, row in enumerate(ROWS):
            factors = [
                2.0 if k == 'SPACE' else
                1.25 if k in ('ENTER', '⌫') else
                1.0 for k in row
            ]
            unit = min((cw - 16) / sum(factors), 260)
            x = (cw - sum(factors) * unit) / 2
            y = keyboard_top + ri * rh
            for k, f in zip(row, factors):
                w = unit * f - gap
                self.rects[k] = (x, y, x + w, y + kh)
                self.centers[k] = (x + w / 2.0, y + kh / 2.0)
                self.row_of[k] = ri
                x += unit * f

    def _build_canvas(self):
        self.canvas.delete('all')
        self.key_items.clear()
        self.key_text_items.clear()
        self.key_fill.clear()
        self.sug_items.clear()
        fs = max(14, min(34, int(self.key_h * 0.25)))
        suggestion_fs = max(12, min(22, int(self.suggestion_h * 0.28)))
        for k, (x1, y1, x2, y2) in self.rects.items():
            is_sug = k in self.sug_keys
            fill = '#1d4ed8' if is_sug and self.sug_cur[self.sug_keys.index(k)] else '#374151'
            rid = self.canvas.create_rectangle(
                x1, y1, x2, y2, fill=fill, outline='#9ca3af', width=2)
            if is_sug:
                label = self.sug_label_text(k)
                font = ('Arial', suggestion_fs, 'bold')
            else:
                label = k
                font = ('Arial', fs, 'bold')
            tid = self.canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2, text=label,
                fill='white', font=font,
                width=(x2 - x1 - 16) if is_sug else 0)
            self.key_items[k] = rid
            self.key_text_items[k] = tid
            self.key_fill[k] = fill
            if is_sug:
                self.sug_items[k] = tid
        self.bar_item = self.canvas.create_rectangle(
            0, 0, 0, 0, fill='#22c55e', outline='', state='hidden')
        r = 11
        self.cursor_item = self.canvas.create_oval(
            -r, -r, r, r, fill='#facc15', outline='#111827', width=2)

    def _refresh_origin(self):
        self.root.update_idletasks()
        self.ox = self.canvas.winfo_rootx()
        self.oy = self.canvas.winfo_rooty()

    def _set_fill(self, k, color):
        if k and k in self.key_items and self.key_fill.get(k) != color:
            self.canvas.itemconfigure(self.key_items[k], fill=color)
            self.key_fill[k] = color

    # ── key hit test (distance to RECTANGLE, not centre) ───────────────────
    def _dist_to_rect(self, k, cx, cy):
        x1, y1, x2, y2 = self.rects[k]
        dx = max(x1 - cx, 0.0, cx - x2)
        dy = max(y1 - cy, 0.0, cy - y2)
        return math.hypot(dx, dy)

    def _nearest_key(self, cx, cy):
        """Return (key, dist-to-rectangle); dist is 0 when inside a key."""
        best_k, best_d = None, float('inf')
        for k in self.rects:
            d = self._dist_to_rect(k, cx, cy)
            if d < best_d:
                best_d, best_k = d, k
        return best_k, best_d

    # ── state helpers ──────────────────────────────────────────────────────
    def _clear_dwell(self):
        self.active         = None
        self.dwell_accum_ms = 0.0
        self.last_dwell_t   = None
        self.repeating      = False
        self.out_since      = None
        self.dwell_pts.clear()

    def _reset_tracking(self):
        self._clear_dwell()
        self.candidate_key = None
        self.median.reset()
        self.fx.reset()
        self.fy.reset()
        self.pos_hist.clear()
        self.dz.reset()
        self.cx = self.cy = None

    def _reset_bias(self):
        self.bias = [0.0, 0.0]
        self.last_bias_delta = None

    # ── implicit drift correction ──────────────────────────────────────────
    def _learn_bias(self, k, now):
        if not self.a.drift_correct:
            return
        if not (len(k) == 1 and k.isalpha()):      # letters only
            return
        if len(self.dwell_pts) < 5:
            return
        pts = np.asarray(self.dwell_pts, dtype=float)
        mx, my = pts.mean(axis=0)
        kx, ky = self.centers[k]
        ex, ey = kx - mx, ky - my
        # If the dwell was far from the key centre the selection is suspect.
        if math.hypot(ex, ey) > 0.75 * self.key_h:
            return
        old = list(self.bias)
        nbx = self.bias[0] + self.a.drift_lr * ex
        nby = self.bias[1] + self.a.drift_lr * ey
        n = math.hypot(nbx, nby)
        if n > self.a.max_bias_px:
            nbx *= self.a.max_bias_px / n
            nby *= self.a.max_bias_px / n
        self.bias = [nbx, nby]
        self.last_bias_delta = (nbx - old[0], nby - old[1])
        self.last_bias_t = now

    def _undo_bias(self, now):
        if self.last_bias_delta is None:
            return
        if now - self.last_bias_t <= UNDO_WINDOW_S:
            self.bias[0] -= self.last_bias_delta[0]
            self.bias[1] -= self.last_bias_delta[1]
        self.last_bias_delta = None

    # ── [AC] autocomplete ──────────────────────────────────────────────────
    def sug_label_text(self, k):
        i = self.sug_keys.index(k)
        s = self.sug_cur[i] if i < len(self.sug_cur) else None
        return s.label if s else ''

    def _apply_suggestions(self, lst, hint=''):
        lab = lambda s: s.label if s else ''
        old = [lab(s) for s in self.sug_cur]
        self.sug_cur = list(lst)
        # if the slot being dwelled on changes under the user, restart its dwell
        if self.active in self.sug_keys:
            i = self.sug_keys.index(self.active)
            if i < len(old) and old[i] != lab(self.sug_cur[i]):
                self._clear_dwell()
        for k, tid in self.sug_items.items():
            self.canvas.itemconfigure(tid, text=self.sug_label_text(k))
            i = self.sug_keys.index(k)
            self._set_fill(k, '#1d4ed8' if i < len(self.sug_cur) and self.sug_cur[i] else '#374151')

    def _refresh_suggestions(self):
        if self.sug_keys:
            self._apply_suggestions(*self.sug.update(
                self.output.get('1.0', 'end-1c')))

    def _prefetch_suggestions(self, text):
        """Hook (the study runner overrides it): speculative LLM request for `text`."""
        fn = getattr(self.sug, 'prefetch', None)   # older autocomplete files lack it
        if fn is not None:
            fn(text)

    def _accept_suggestion(self, k):
        i = self.sug_keys.index(k)
        s = self.sug_cur[i] if i < len(self.sug_cur) else None
        if s is None or s.insert is None:          # empty / "thinking" slot
            return
        text = self.output.get('1.0', 'end-1c')
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', text[:len(text) - s.replace_len] + s.insert)

    # ── selection ──────────────────────────────────────────────────────────
    def _select(self, k, conf, sx, sy, cx, cy, elapsed, threshold):
        now = time.perf_counter()
        if k.startswith(SUG_PREFIX):                                # [AC]
            self._accept_suggestion(k)
        elif k == '⌫':
            s = self.output.get('1.0', 'end-1c')
            self.output.delete('1.0', 'end')
            self.output.insert('1.0', s[:-1])
            self._undo_bias(now)
        else:
            if k == 'SPACE':
                self.output.insert('end', ' ')
            elif k == 'ENTER':
                self.output.insert('end', '\n')
            else:
                self.output.insert('end', k)
            self._learn_bias(k, now)
        self._refresh_suggestions()                                 # [AC]
        self.last_select_t = now
        self._n_sel += 1
        if k == '⌫':
            self.cascade_n = 0                    # slow down again after a correction
        elif not k.startswith(SUG_PREFIX):
            self.cascade_n += 1
        self.flash_key   = k
        self.flash_until = now + 0.35
        self.log_row('selected', gaze_key=k, selected=k, conf=conf,
                     sx=sx, sy=sy, cx=cx, cy=cy,
                     dwell=elapsed, threshold=threshold)

    # ── saccade detection (speed over ~3 frames, on filtered position) ─────
    def _is_saccade(self):
        if len(self.pos_hist) < 3:
            return False
        t0, x0, y0 = self.pos_hist[0]
        t1, x1, y1 = self.pos_hist[-1]
        dt = t1 - t0
        if dt < 0.02:
            return False
        return math.hypot(x1 - x0, y1 - y0) / dt > self.a.saccade_threshold

    # ── main step ──────────────────────────────────────────────────────────
    def step(self, sx, sy, conf, now):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        if (cw, ch) != self.layout_size:
            self._layout(cw, ch)
            self._build_canvas()
            self.layout_size = (cw, ch)
            self._refresh_origin()
            self._reset_tracking()
            self._reset_bias()
        elif self.frame_count % 20 == 0:
            self._refresh_origin()

        # Map calibration -> canvas pixels
        raw_cx = sx * self.screen_w / max(1.0, self.cal_w) - self.ox
        raw_cy = sy * self.screen_h / max(1.0, self.cal_h) - self.oy

        conf_ok = conf >= self.a.confidence_threshold

        # Low-confidence frames are NOT fed to the filters: hold last position.
        if conf_ok or self.cx is None:
            mx, my = self.median.update(raw_cx, raw_cy)
            fx = self.fx.update(mx, now)
            fy = self.fy.update(my, now)
            self.pos_hist.append((now, fx, fy))
            is_saccade = self._is_saccade()
            fx, fy = self.dz.update(fx, fy)       # stabilise AFTER saccade detection
            cx = min(max(fx + self.bias[0], 0.0), float(cw))
            cy = min(max(fy + self.bias[1], 0.0), float(ch))
            self.cx, self.cy = cx, cy
        else:
            cx, cy = self.cx, self.cy
            is_saccade = False

        # ── Key under gaze (rectangle distance) ─────────────────────────
        nk, nd = self._nearest_key(cx, cy)
        gaze_key = nk if nd <= self.a.snap_px else None

        # Suggestion magnet: the slots sit at the very top of the screen, where gaze
        # estimates tend to fall short. A slot that currently HAS a word is selectable
        # from up to `sug_reach_px` away (empty slots never attract the cursor).
        if self.a.sug_reach_px > 0 and not (gaze_key or '').startswith(SUG_PREFIX):
            best_k, best_d = None, self.a.sug_reach_px
            for i, sk in enumerate(self.sug_keys):
                if self.sug_cur[i] is None:
                    continue
                d = self._dist_to_rect(sk, cx, cy)
                if d <= best_d:
                    best_k, best_d = sk, d
            if best_k is not None:
                gaze_key = best_k

        # ── Hysteresis on the active key ────────────────────────────────
        if self.active is not None:
            d_act = self._dist_to_rect(self.active, cx, cy)
            # A populated suggestion slot that is attracting the cursor releases the
            # held letter key, even if the cursor still touches that key's edge.
            magnet_switch = (gaze_key is not None and gaze_key.startswith(SUG_PREFIX)
                             and gaze_key != self.active
                             and not self.active.startswith(SUG_PREFIX))
            hold_px = self.a.hold_px
            if self.active.startswith(SUG_PREFIX):      # slots keep their reach radius
                hold_px = max(hold_px, self.a.sug_reach_px)
            if d_act <= hold_px and not magnet_switch:
                self.out_since = None
            else:
                if self.out_since is None:
                    self.out_since = now
                if (now - self.out_since) * 1000.0 >= self.a.leave_ms:
                    self.log_row('dwell_reset', gaze_key=gaze_key, conf=conf,
                                 sx=sx, sy=sy, cx=cx, cy=cy,
                                 dwell=self.dwell_accum_ms)
                    self._clear_dwell()
        holding = self.active is not None and self.out_since is None

        # ── Candidate debounce (only for ACTIVATING a new key) ──────────
        if gaze_key != self.candidate_key:
            self.candidate_key   = gaze_key
            self.candidate_since = now

        stable_enough = (now - self.candidate_since) * 1000 >= self.a.stability_ms

        if self.active is None and gaze_key is not None and stable_enough:
            if (now - self.last_select_t) * 1000.0 > self.a.cascade_reset_ms:
                self.cascade_n = 0                # typing paused: start slow again
            self.active         = gaze_key
            self.dwell_accum_ms = 0.0
            self.last_dwell_t   = None
            self.repeating      = False
            self.out_since      = None
            self.dwell_pts.clear()
            holding = True

        key = self.active

        # ── Dwell accumulation ──────────────────────────────────────────
        if self.last_dwell_t is None:
            dt_ms = 0.0
        else:
            dt_ms = min((now - self.last_dwell_t) * 1000.0, MAX_DWELL_DT_MS)
        self.last_dwell_t = now

        if key and holding and not is_saccade and conf_ok:
            self.dwell_accum_ms += dt_ms
            self.dwell_pts.append((cx, cy))
        elif is_saccade:
            # Decay 2x faster than accumulation — clears quickly on movement
            self.dwell_accum_ms = max(0.0, self.dwell_accum_ms - dt_ms * 2.0)
        # Low confidence / briefly outside the key -> pause

        # ── Threshold & fire ────────────────────────────────────────────
        if self.a.mode == 'fixed':
            threshold = self.a.dwell_ms
        else:
            threshold = self.a.fast_ms if conf_ok else self.a.slow_ms
        if (self.a.cascade_step_ms > 0 and key and key not in ('⌫', 'ENTER')
                and not key.startswith(SUG_PREFIX)):
            floor = min(threshold, self.a.cascade_min_ms)
            threshold = max(floor, threshold - self.cascade_n * self.a.cascade_step_ms)
        if self.repeating:
            threshold = max(threshold, self.a.repeat_ms)

        rearm_ok = (now - self.last_select_t) >= self.a.rearm_ms / 1000.0

        # Latency hiding: while the user is still dwelling on a key, start the LLM
        # request for the text that key WILL produce (cached before the selection).
        if (self.a.prefetch_ms >= 0 and key and holding and conf_ok
                and key not in ('⌫', 'ENTER') and not key.startswith(SUG_PREFIX)
                and self.dwell_accum_ms >= self.a.prefetch_ms
                and self._pf_for != (key, self._n_sel)):
            self._pf_for = (key, self._n_sel)
            self._prefetch_suggestions(
                self.output.get('1.0', 'end-1c') + (' ' if key == 'SPACE' else key))

        if key and holding and self.dwell_accum_ms >= threshold and rearm_ok:
            self._select(key, conf, sx, sy, cx, cy, self.dwell_accum_ms, threshold)
            self.dwell_accum_ms = 0.0
            self.dwell_pts.clear()
            self.repeating      = True

        # ── Rendering ───────────────────────────────────────────────────
        wanted = {}
        if key:
            wanted[key] = '#2563eb'
        if self.flash_key and now < self.flash_until:
            wanted[self.flash_key] = '#16a34a'

        for k in self.key_fill:
            default_col = '#1d4ed8' if k in self.sug_keys and self.sug_cur[self.sug_keys.index(k)] else '#374151'
            col = wanted.get(k, default_col)
            if self.key_fill[k] != col:
                self._set_fill(k, col)

        r = 11
        self.canvas.coords(self.cursor_item, cx - r, cy - r, cx + r, cy + r)
        self.canvas.tag_raise(self.cursor_item)

        if key and self.dwell_accum_ms > 0.0:
            x1, y1, x2, y2 = self.rects[key]
            frac = min(1.0, self.dwell_accum_ms / max(1.0, threshold))
            self.canvas.coords(
                self.bar_item,
                x1 + 4, y2 - 10,
                x1 + 4 + (x2 - x1 - 8) * frac, y2 - 4,
            )
            self.canvas.itemconfigure(self.bar_item, state='normal')
            self.canvas.tag_raise(self.bar_item)
        else:
            self.canvas.itemconfigure(self.bar_item, state='hidden')

        if self.frame_count % self.a.status_every == 0:
            self.status.config(text=(
                f'{self.a.mode.upper()} | '
                f'gaze=({sx:.0f},{sy:.0f}) | '
                f'cursor=({cx:.0f},{cy:.0f}) | '
                f'key={key or "—"} | '
                f'conf={conf:.2f} | '
                f'fps={self._fps():.0f} | '
                f'dwell={self.dwell_accum_ms:.0f}/{threshold:.0f}ms | '
                f'bias=({self.bias[0]:+.0f},{self.bias[1]:+.0f}) | '
                f'saccade={is_saccade} | F5 reset bias | ESC exits'
            ))

    def _fps(self):
        if len(self.res_times) < 2:
            return 0.0
        span = self.res_times[-1] - self.res_times[0]
        return (len(self.res_times) - 1) / span if span > 0 else 0.0

    def on_lost(self, err, now):
        self.status.config(text=f'No face: {err}')
        if self.lost_since is None:
            self.lost_since = now
        elif now - self.lost_since > 0.5:
            self._reset_tracking()

    def loop(self):
        if self.sug_keys:                                           # [AC]
            r = self.sug.poll()
            if r is not None:
                self._apply_suggestions(*r)
        res = self.tracker.take()
        if res is not None:
            self.frame_count += 1
            r, err, now = res
            self.res_times.append(now)
            try:
                if r is None:
                    self.on_lost(err, now)
                else:
                    self.lost_since = None
                    self.step(r[0], r[1], r[2], now)
            except Exception as e:
                self.status.config(text=f'Error: {e}')
        self.root.after(self.a.ui_ms, self.loop)

    def close(self):
        self.tracker.running = False
        self.grabber.running = False
        try:
            self.log.close()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────── CLI
def build_parser(add_help=True):
    """All CLI options. Reused by the Phase A/B/C study runner so both always agree."""
    p = argparse.ArgumentParser(description='Gaze virtual keyboard v4',
                                add_help=add_help)

    p.add_argument('--calibration', '--calibration-csv', dest='calibration',
                   required=True)
    p.add_argument('--model',       default='models/L2CSNet_mpiigaze.pkl')
    p.add_argument('--camera',      type=int,   default=0)
    p.add_argument('--face-size',   type=int,   default=448)
    p.add_argument('--cuda',        action='store_true')
    p.add_argument('--fullscreen',  action='store_true')
    p.add_argument('--output',      default='interaction_data/dwell_events.csv')

    # Smoothing
    p.add_argument('--min-cutoff', type=float, default=1.0,
                   help='One Euro min cutoff (Hz). Lower = steadier on a key, '
                        'slower start. Try 0.8-2.0.')
    p.add_argument('--beta',       type=float, default=0.004,
                   help='One Euro speed coefficient. Higher = less lag on fast '
                        'moves, more jitter. Try 0.003-0.02.')
    p.add_argument('--median-n',   type=int,   default=3,
                   help='Raw-sample median window (spike/blink rejection).')

    p.add_argument('--deadzone-px', type=float, default=20,
                   help='Cursor dead-zone radius (px). Output ignores movement '
                        'smaller than this. 0 disables. Try 15-30.')
    p.add_argument('--no-angle-smooth', dest='angle_smooth', action='store_false',
                   help='Disable smoothing of gaze/head angles before the mapper.')
    p.add_argument('--angle-min-cutoff', type=float, default=2.0,
                   help='One Euro min cutoff (Hz) for the angle inputs. Lower = '
                        'steadier, more lag. Try 1.0-3.0.')
    p.add_argument('--angle-beta', type=float, default=0.05,
                   help='One Euro speed coefficient for the angle inputs (deg/s).')

    # Dwell
    p.add_argument('--mode',     choices=['fixed', 'adaptive'], default='fixed')
    p.add_argument('--dwell-ms', type=float, default=1200,
                   help='Fixed dwell time ms. 1200 is safe first value.')
    p.add_argument('--fast-ms',  type=float, default=900)
    p.add_argument('--slow-ms',  type=float, default=1500)
    p.add_argument('--confidence-threshold', type=float, default=0.35)
    p.add_argument('--stability-ms', type=float, default=200,
                   help='ms gaze must stay on new key before it activates.')
    p.add_argument('--rearm-ms',     type=float, default=800,
                   help='Lockout ms after selection. Must be < dwell-ms.')
    p.add_argument('--repeat-ms',    type=float, default=1500)
    p.add_argument('--cascade-step-ms', type=float, default=0,
                   help='Cascading dwell: each consecutive letter/space selection '
                        'shortens the next dwell by this many ms (0 = off). It resets '
                        'on a backspace or a pause. Suggestion slots, backspace and '
                        'enter always use the full dwell. Try 100.')
    p.add_argument('--cascade-min-ms', type=float, default=450,
                   help='Shortest dwell the cascade may reach.')
    p.add_argument('--cascade-reset-ms', type=float, default=2500,
                   help='A pause longer than this before the next key resets the cascade.')
    p.add_argument('--sug-reach-px', type=float, default=30,
                   help='A suggestion slot that shows a word can be selected from this '
                        'many px away (helps when the cursor stops short of the top '
                        'strip). 0 = off. Try 20-50.')
    p.add_argument('--prefetch-ms', type=float, default=200,
                   help='Start the LLM request for a key after dwelling on it this long, '
                        'so suggestions are ready when it is selected. -1 disables.')

    # Key lookup / hysteresis
    p.add_argument('--snap-px', type=float, default=15,
                   help='Max px outside a key rectangle that still counts as on-key.')
    p.add_argument('--hold-px', type=float, default=5,
                   help='Active key stays held while cursor is within this many '
                        'px of its rectangle.')
    p.add_argument('--leave-ms', type=float, default=100,
                   help='ms the cursor must stay out of the active key before '
                        'dwell is reset (dwell pauses meanwhile).')

    # Saccade
    p.add_argument('--saccade-threshold', type=float, default=600,
                   help='px/sec (filtered position, ~3 frame window) that counts '
                        'as a saccade and pauses dwell.')

    # Drift correction
    p.add_argument('--no-drift-correct', dest='drift_correct',
                   action='store_false',
                   help='Disable implicit drift correction.')
    p.add_argument('--drift-lr', type=float, default=0.12,
                   help='Fraction of the observed offset absorbed per selection.')
    p.add_argument('--max-bias-px', type=float, default=60,
                   help='Hard cap on total drift-correction offset.')

    # UI
    p.add_argument('--ui-ms',       type=int, default=8)
    p.add_argument('--status-every',type=int, default=4)

    # Autocomplete [AC]
    p.add_argument('--n-suggestions', type=int, default=3,
                   help='Suggestion slots above the keyboard; must match the '
                        'calibration run. 0 = original layout, feature off.')
    p.add_argument('--llm-url', default='http://localhost:11434/v1/chat/completions',
                   help='OpenAI-compatible chat endpoint used for word prediction '
                        '(default: local Ollama). Pass "" to disable suggestions.')
    p.add_argument('--llm-model', default='gemma2:2b')
    p.add_argument('--llm-timeout', type=float, default=6.0)
    p.add_argument('--llm-hint', default='',
                   help='Optional one-line background about the user/topics, given '
                        'to the model as context, e.g. "A university student writing '
                        'messages to friends and family."')

    return p


def main():
    a = build_parser().parse_args()
    a.calibration = str(Path(a.calibration).resolve())
    a.model       = str(Path(a.model).resolve())
    App(a).run()


if __name__ == '__main__':
    main()