"""autocomplete_1.py - LLM-only, context-aware word prediction for the gaze keyboard.

Two modes, both driven by the text already in the box (topic, grammar, tone):
  * COMPLETE  - mid-word ("i am feeling very t"): proposes full words, including
                other forms of a word that is already complete (complete -> completed).
  * NEXT WORD - after a space ("i am feeling very "), or at the start of a line:
                proposes the next word.
There is NO offline word list: every suggestion comes from the model.

The LLM runs in a background thread (latest request wins), so the keyboard,
gaze tracking and dwell logic never wait on it. Standard library only.

Speed notes
  * ONE shared system prompt + ONE shared set of examples is used for both modes,
    so every request starts with the same tokens and the server (Ollama/llama.cpp)
    can reuse its cached prompt prefix: only the last ~30 tokens are processed.
  * The model answers with a plain comma-separated line (no JSON), stops at the
    newline, and is limited to a few tokens, so generation is short.

Set AUTOCOMPLETE_DEBUG=1 to print every request, its latency and the raw words.
"""
import json
import os
import re
import threading
import time
import urllib.request
from collections import deque, namedtuple

SUG_PREFIX = 'SUG'          # must match the calibration script
DEBUG = bool(os.environ.get('AUTOCOMPLETE_DEBUG'))
WARMUP_TIMEOUT = 90.0       # first call may have to load the model into memory
CONTEXT_CHARS = 160         # only the current line, last ~160 chars, is sent

# label: text shown on the slot   replace_len: typed chars it replaces
# insert: text put in their place (None = not selectable)
Suggestion = namedtuple('Suggestion', 'label replace_len insert')

_WORD_RE = re.compile(r'([A-Za-z]+)\Z')

NEXT_MARK = '(none - predict the next word)'

SYSTEM = """\
You are the word-prediction engine of an eye-gaze keyboard. The user types one \
letter at a time by looking at keys, so every letter is slow. Each correct \
suggestion lets them enter a whole word with ONE glance, so you decide how fast \
they can write.

INPUT: the text typed so far, and the PARTIAL WORD being typed right now. If the \
partial word is "{next}", the user just pressed space (or is starting a line) and \
needs the NEXT word.
OUTPUT: ONE line only: up to {k} likely words, lowercase, separated by commas, most \
likely first. No explanation, no numbering, no quotes.

HOW TO PREDICT
1. Act like the language model of a phone keyboard: output the most PROBABLE words, \
not the most interesting ones. Ordinary writing is dominated by short function \
words, so they MUST appear whenever they fit: the, a, an, and, of, to, in, is, it, \
you, that, for, on, with, as, are, was, this, but, be, have, not, at, my, we, i, \
they, he, she, can, will, if, or, so, do, what, there, not. Do not skip them just \
because they are short or common; usually put them FIRST when the grammar allows.
2. Use the text so far: grammar (after "to" a base verb; after "the" a noun or \
adjective; after "is" an adjective, an -ing verb, "a", "the" or "not"), topic, \
situation and tone. Choose words that fit THIS sentence.
3. If a partial word is given, EVERY word must start with exactly those letters and \
be LONGER than them (never repeat the partial word). The partial word may already be \
a whole word; then offer its other forms (complete -> completed, completely).
4. Offer the forms of a word together and let the sentence pick the order: ring, \
rings; walk, walked, walking. Singular after "a"/"one"/"this", plural after \
"two"/"many"/"these", -ing after "was"/"are", past after "have"/"had".
5. Any topic is possible (chat, work, study, technical, creative). Infer it from the \
text; if there is no clue, use the most common everyday English words.
6. Words only: letters a-z, no apostrophes, digits, spaces or punctuation inside a \
word. Never repeat the word just before the cursor."""

# (text so far, partial word or None for next-word, answer words)
_EXAMPLES = [
    ('', 't', ['the', 'to', 'that', 'this', 'they', 'there']),
    ('', None, ['the', 'i', 'we', 'it', 'my', 'this']),
    ('what is', 't', ['the', 'that', 'this', 'there', 'to', 'today']),
    ('what is', None, ['the', 'your', 'that', 'this', 'a', 'it']),
    ('i', 'a', ['am', 'always', 'also', 'ate', 'around', 'asked']),
    ('this is', 'i', ['it', 'important', 'in', 'interesting', 'impossible', 'if']),
    ('i want to', None, ['go', 'be', 'see', 'have', 'get', 'know']),
    ('he went to the', 's', ['store', 'school', 'shop', 'station', 'street', 'sea']),
    ('he went to', None, ['the', 'a', 'school', 'work', 'see', 'bed']),
    ('thank you', None, ['for', 'very', 'so', 'again', 'all', 'much']),
    ('i am going', None, ['to', 'home', 'back', 'out', 'there', 'now']),
    ('please turn off the', 'l', ['light', 'lights', 'lamp', 'laptop', 'lock', 'lamps']),
    ('i have', 'complete', ['completed', 'completely', 'completes', 'completeness']),
    ('i bought two', 'ring', ['rings', 'ringtones', 'ringers']),
    ('they were', 'walk', ['walking', 'walked', 'walkers']),
    ('give me a', None, ['call', 'chance', 'break', 'moment', 'hand', 'minute']),
    ('one of the', None, ['best', 'most', 'people', 'things', 'first', 'main']),
    ('where did i leave my', 'k', ['keys', 'key', 'keyboard', 'kitchen']),
]


_FORM_SUFFIXES = ('s', 'es', 'ed', 'ing', 'ly', 'er', 'ers')


def _is_form(a, b):
    """True if a and b look like inflected forms of the same word (ring/rings,
    walk/walking, study/studies, make/making, run/running...)."""
    if a == b:
        return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    if len(short) < 3 or not long_.startswith(short[:-1]):
        return False
    if long_.startswith(short):
        rest = long_[len(short):]
        if rest in _FORM_SUFFIXES:
            return True
        if short.endswith('e') and rest == 'd':                  # free -> freed
            return True
        if rest[:1] == short[-1:] and rest[1:] in ('ing', 'ed', 'er', 'ers', 'y'):
            return True                                          # run -> running
    if short.endswith('y') and long_[len(short) - 1:] in ('ies', 'ied', 'ier', 'ily'):
        return True                                              # study -> studies
    if short.endswith('e') and long_[len(short) - 1:] in ('ing', 'ings'):
        return True                                              # make -> making
    return False


def _group_forms(words):
    """Keep forms of the same word adjacent, best word first:
    [ring, rink, rings] -> [ring, rings, rink]."""
    rest, out = list(words), []
    while rest:
        w = rest.pop(0)
        out.append(w)
        family = [x for x in rest if _is_form(w, x)]
        for x in family:
            rest.remove(x)
            out.append(x)
    return out


def _user_msg(context, prefix):
    return (f'Text so far: {context or "(none)"}\n'
            f'Partial word: {prefix or NEXT_MARK}')


def _parse_words(txt, prefix, context):
    """Pull the words out of the model's reply (comma list, JSON array, or
    anything in between) and validate them."""
    line = txt.strip().split('\n')[0]
    if ':' in line:                       # "Here are the words: a, b, c"
        line = line.split(':', 1)[1]
    last = context.split()[-1] if context.split() else ''
    out = []
    for w in re.findall(r'[a-z]+', line.lower()):
        # hard validation: one alphabetic word that really extends the prefix
        # (prefix is '' in next-word mode, so any alphabetic word passes)
        if w.startswith(prefix) and len(w) > len(prefix) and w not in out:
            if not prefix and w == last:  # never suggest the word just typed
                continue
            out.append(w)
    return out


class _LLMWorker(threading.Thread):
    """Latest-request-wins background worker for an OpenAI-compatible API."""

    def __init__(self, url, model, timeout, n, hint=''):
        super().__init__(daemon=True)
        self.url, self.model, self.timeout, self.n = url, model, timeout, n
        self.hint = (hint or '').strip()
        self.k = n + 1              # ask for n words + 1 spare (in case one fails
                                    # validation); the best n are shown
        self.cv = threading.Condition()
        self.job = None             # real request (what the user needs NOW)
        self.pre_job = None         # speculative request (lower priority)
        self.busy = None            # key being asked right now
        self.out = deque()          # every finished result; poll() drains all
        self.ready = threading.Event()   # set once warm-up has finished (or failed)
        self.start()

    def submit(self, key, context, prefix, prefetch=False):
        with self.cv:
            if prefetch:
                self.pre_job = (key, context, prefix)      # latest prefetch wins
            else:
                self.job = (key, context, prefix)           # latest real request wins
                if self.pre_job and self.pre_job[0] == key:  # real one supersedes it
                    self.pre_job = None
            self.cv.notify()

    def is_pending(self, key):
        """True if `key` is being asked right now or is queued as a real request."""
        with self.cv:
            return key == self.busy or (self.job is not None and self.job[0] == key)

    def take(self):
        with self.cv:
            r = list(self.out)
            self.out.clear()
            return r

    def run(self):
        # Warm-up: loads the model AND the shared prompt prefix into the server's
        # cache before the user types, so the first real request is fast.
        try:
            t = time.perf_counter()
            self._ask('', 'h', WARMUP_TIMEOUT)
            print(f'[autocomplete] LLM ready ({time.perf_counter() - t:.1f}s warm-up)')
        except Exception as e:
            print(f'[autocomplete] LLM warm-up failed: {e}\n'
                  f'               Check the server is running and the model name '
                  f'is right ({self.url}, model "{self.model}").')
        self.ready.set()
        while True:
            with self.cv:
                while self.job is None and self.pre_job is None:
                    self.cv.wait()
                if self.job is not None:                    # real requests go first
                    key, context, prefix = self.job
                    self.job = None
                else:
                    key, context, prefix = self.pre_job
                    self.pre_job = None
                self.busy = key
            try:
                t = time.perf_counter()
                words = self._ask(context, prefix, self.timeout)
                if DEBUG:
                    print(f'[autocomplete] "{context}" + "{prefix}" -> {words} '
                          f'({time.perf_counter() - t:.2f}s)')
            except Exception as e:                       # network, timeout, parse...
                print(f'[autocomplete] LLM error: {e}')
                words = None                             # None = failed (not cached)
            with self.cv:
                self.out.append((key, words))
                self.busy = None

    def _messages(self, context, prefix):
        system = SYSTEM.format(k=self.k, next=NEXT_MARK)
        if self.hint:
            system += ('\n\nBackground about the user (use it to judge topic and '
                       'vocabulary): ' + self.hint)
        msgs = [{'role': 'system', 'content': system}]
        for ctx, pre, words in _EXAMPLES:
            msgs.append({'role': 'user', 'content': _user_msg(ctx, pre)})
            msgs.append({'role': 'assistant',
                         'content': ', '.join(words[:self.k])})
        msgs.append({'role': 'user', 'content': _user_msg(context, prefix)})
        return msgs

    def _ask(self, context, prefix, timeout):
        body = {
            'model': self.model,
            'messages': self._messages(context, prefix),
            'temperature': 0.0,
            'max_tokens': 5 * self.k,  # safety ceiling only (~3-4 tokens per word);
            'stop': ['\n'],           # the reply normally ends itself after k words
        }
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('OPENAI_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
        req = urllib.request.Request(
            self.url, json.dumps(body).encode('utf-8'), headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            choice = json.load(r)['choices'][0]
        txt = choice['message']['content']
        words = _parse_words(txt, prefix, context)
        if choice.get('finish_reason') == 'length' and words:
            words.pop()              # reply was cut off: last word may be partial
        return _group_forms(words)[:self.k]


class Suggester:
    def __init__(self, n=3, llm_url=None, llm_model='gemma2:2b',
                 llm_timeout=6.0, upper=True, hint=''):
        self.n = n
        self.upper = upper
        self.llm = (_LLMWorker(llm_url, llm_model, llm_timeout, n, hint)
                    if llm_url and n > 0 else None)
        self._key = None          # (context, prefix) currently on screen
        self._words = []          # words currently shown
        self._cache = {}

    def is_ready(self):
        """True once the LLM has finished warming up (or suggestions are off)."""
        return self.llm is None or self.llm.ready.is_set()

    # ------------------------------------------------------------ helpers
    def _case(self, s):
        return s.upper() if self.upper else s

    def _slots(self, words, rlen):
        sugs = [Suggestion(self._case(w), rlen, self._case(w) + ' ')
                for w in words[:self.n]]
        return sugs + [None] * (self.n - len(sugs))

    @staticmethod
    def _line(s):
        """Current line only (earlier lines are separate sentences), trimmed."""
        return s.split('\n')[-1].strip()[-CONTEXT_CHARS:].lower()

    @classmethod
    def _parse(cls, text):
        """text -> (context, partial_word). partial_word '' = NEXT-WORD mode.
        Also predicts the first word of a new line."""
        m = _WORD_RE.search(text)
        if m:                                      # mid-word -> COMPLETE mode
            return cls._line(text[:m.start()]), m.group(1).lower()
        return cls._line(text), ''                 # after a space / new line -> NEXT

    def prefetch(self, text):
        """Start the LLM request for `text` in the background WITHOUT touching what is
        shown. Call it with the text a key WILL produce while the user is still
        dwelling on it, so the answer is cached by the time the selection happens."""
        if not self.llm:
            return
        key = self._parse(text)
        if key in self._cache or self.llm.is_pending(key):
            return
        self.llm.submit(key, key[0], key[1], prefetch=True)

    # ------------------------------------------------------------ main API
    def update(self, text):
        """Call whenever the text changes.  Returns (slots, hint)."""
        if not self.llm:
            return self._slots([], 0), ''
        context, pre = self._parse(text)
        key = (context, pre)
        self._key = key

        if key in self._cache:                     # already asked this before
            self._words = self._cache[key]
            return self._slots(self._words, len(pre)), ''

        # While the LLM thinks, keep earlier suggestions that still fit the
        # longer prefix (e.g. HELLO/HELP stay valid when "H" becomes "HE").
        if pre:
            self._words = [w for w in self._words
                           if w.startswith(pre) and len(w) > len(pre)]
        else:                                      # old words belonged to another position
            self._words = []
        if not self.llm.is_pending(key):           # already in flight (e.g. prefetched)
            self.llm.submit(key, context, pre)
        return (self._slots(self._words, len(pre)),
                f'Predicting “{pre.upper()}”…' if pre else 'Predicting next word…')

    def poll(self):
        """Non-blocking. Returns (slots, hint) when an LLM answer for the
        CURRENT text has arrived, else None."""
        if not self.llm:
            return None
        results = self.llm.take()
        out = None
        for key, words in results:
            if words:                              # cache real answers only,
                if len(self._cache) > 500:         # never failures / empties
                    self._cache.clear()
                self._cache[key] = words
            if key != self._key:                   # user typed on; stale
                continue
            pre = key[1]
            if words is None:                      # request failed
                out = (self._slots(self._words, len(pre)),
                       'Suggestions unavailable (is the LLM server running?)')
            else:
                self._words = words
                out = (self._slots(words, len(pre)),
                       (f'Completing “{pre.upper()}”' if pre else 'Next word')
                       if words else '')
        return out