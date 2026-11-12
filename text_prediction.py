
from spellchecker import SpellChecker

spell = SpellChecker()


def autocomplete(prefix, limit=3):
    """Return up to `limit` dictionary words beginning with prefix."""
    prefix = prefix.strip().lower()

    if not prefix:
        return []

    dictionary = spell.word_frequency.dictionary

    matches = [
        word for word in dictionary
        if word.startswith(prefix)
    ]

    matches.sort(
        key=lambda word: dictionary[word],
        reverse=True
    )

    return matches[:max(0, int(limit))]


def autocorrect(word):
    """Correct an unknown word; leave known words unchanged."""
    original = word.strip()

    if not original:
        return original

    lower = original.lower()

    if spell.known([lower]):
        return original

    corrected = spell.correction(lower)

    if not corrected:
        return original

    # Preserve simple sentence-initial capitalization.
    if original[:1].isupper():
        corrected = corrected.capitalize()

    return corrected