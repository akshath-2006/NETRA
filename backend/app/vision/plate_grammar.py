"""Indian number plate grammar: validation, repair and confusion-aware similarity.

THIS FILE IS THE PROJECT'S CHEAPEST BIG WIN. About two hundred lines, no model,
no training, and it does more for end-to-end accuracy than swapping OCR engines
would.

Three ideas, in order of value:

**1. The format is a constraint, so use it.**
An Indian plate is ``AA 00 A(A)(A) 0000`` -- two state letters, one or two
district digits, one to three series letters, four digits. If OCR returns "0"
where a letter must be, it is almost certainly "O". Repairing by position turns
a large slice of near-misses into exact hits, for free.

**2. OCR errors are not random.**
"0" and "O" are confused constantly; "0" and "W" essentially never. A plain
edit distance treats those the same and is wrong to. The weighted distance here
charges 0.25 for a known-confusable substitution and 1.0 for anything else,
which is what lets M7 link MH12DE1433 to a read of MH12OE1433 without also
linking two genuinely different vehicles.

**3. Validity is a signal, not a gate.**
We never throw away an unreadable plate -- we score it. A read that matches the
grammar and has a real state code is worth more than one that does not, and the
association engine gets told which is which.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Registered state / UT codes. A read whose first two characters are not in this
# set is very likely misread -- cheap, strong signal.
STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB",
}

# Standard series:  KA 01 AB 1234
# District is 01-99 -- explicitly NOT 0 or 00, which no real plate carries.
# Being strict here is what lets repair() spot a misread district at all.
STANDARD_RE = re.compile(r"^([A-Z]{2})(0[1-9]|[1-9][0-9]?)([A-Z]{1,3})([0-9]{4})$")
# Bharat series:    21 BH 1234 AB
BH_RE = re.compile(r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$")

# Letters that OCR most often returns when the true character is a digit.
LETTER_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0", "U": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2", "E": "3", "A": "4", "S": "5",
    "G": "6", "C": "6", "T": "7", "Y": "7",
    "B": "8", "R": "8",
}
# ...and the reverse, for positions that must be letters.
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
}

# Pairs a human (or an OCR model) genuinely mixes up. Symmetric.
CONFUSABLE_PAIRS = [
    ("0", "O"), ("0", "D"), ("0", "Q"), ("0", "U"),
    ("1", "I"), ("1", "L"), ("1", "J"), ("I", "L"),
    ("2", "Z"), ("3", "E"), ("4", "A"), ("5", "S"),
    ("6", "G"), ("6", "C"), ("G", "C"),
    ("7", "T"), ("7", "Y"), ("T", "Y"),
    ("8", "B"), ("8", "R"), ("B", "R"),
    ("9", "G"), ("9", "P"),
    # Letter-to-letter confusions. These were missing at first and the tests
    # caught it: MH12DE1433 misread as MH12OE1433 is a D/O swap, which is one
    # of the most common plate misreads there is. Without the pair listed, that
    # near-miss cost a full substitution and scored like a different vehicle.
    ("D", "O"), ("O", "Q"), ("D", "Q"), ("U", "V"), ("V", "W"),
    ("P", "R"), ("I", "J"), ("E", "F"), ("M", "N"), ("K", "X"),
    ("C", "G"), ("S", "Z"),
]
CONFUSABLE: set[frozenset[str]] = {frozenset(p) for p in CONFUSABLE_PAIRS}

CONFUSABLE_COST = 0.25   # cost of swapping a known-confusable pair
SUBSTITUTE_COST = 1.0    # cost of any other substitution
GAP_COST = 1.0           # insertion or deletion


def normalise(text: str) -> str:
    """Strip everything that is not A-Z or 0-9 and upper-case the rest."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlateFormat:
    valid: bool
    kind: str                 # "standard" | "bh" | "unknown"
    state: str = ""
    known_state: bool = False

    @property
    def score(self) -> float:
        """0-1 confidence that this string is a real plate at all."""
        if not self.valid:
            return 0.0
        return 1.0 if self.known_state else 0.75


def inspect(text: str) -> PlateFormat:
    """Does this string look like a real Indian plate?"""
    t = normalise(text)

    m = STANDARD_RE.match(t)
    if m:
        state = m.group(1)
        return PlateFormat(True, "standard", state, state in STATE_CODES)

    m = BH_RE.match(t)
    if m:
        return PlateFormat(True, "bh", "BH", True)

    return PlateFormat(False, "unknown")


def is_valid(text: str) -> bool:
    return inspect(text).valid


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def _coerce(chars: list[str], start: int, end: int, to_digit: bool) -> None:
    """Force a slice towards digits or letters, in place."""
    table = LETTER_TO_DIGIT if to_digit else DIGIT_TO_LETTER
    for i in range(start, min(end, len(chars))):
        c = chars[i]
        wrong_kind = c.isalpha() if to_digit else c.isdigit()
        if wrong_kind and c in table:
            chars[i] = table[c]


def repair(text: str) -> tuple[str, bool]:
    """Nudge a read towards the plate grammar using positional constraints.

    Returns ``(repaired_text, was_changed)``. An already-valid read is returned
    untouched -- we never "fix" something that was right.

    The district block may be one digit or two, so the series length is
    ambiguous from length alone. We try both splits and keep whichever produces
    a valid plate, preferring one with a real state code.
    """
    t = normalise(text)
    if not t or is_valid(t):
        return t, False
    if len(t) not in (9, 10):
        # Too far from the canonical shape to guess at responsibly.
        return t, False

    candidates: list[str] = []
    for district in (2, 1):
        series = len(t) - 2 - district - 4
        if not 1 <= series <= 3:
            continue
        chars = list(t)
        _coerce(chars, 0, 2, to_digit=False)                              # state
        _coerce(chars, 2, 2 + district, to_digit=True)                    # district
        _coerce(chars, 2 + district, 2 + district + series, to_digit=False)  # series
        _coerce(chars, 2 + district + series, len(t), to_digit=True)      # number
        candidate = "".join(chars)
        # Only repair towards a plate that could actually exist. Without the
        # known-state requirement this happily converts noise into a
        # structurally valid plate with an invented state code -- which is
        # exactly the kind of confident nonsense this project exists to avoid.
        if candidate != t and inspect(candidate).known_state:
            candidates.append(candidate)

    if not candidates:
        return t, False
    # A candidate with a recognised state code beats one without.
    candidates.sort(key=lambda c: inspect(c).score, reverse=True)
    return candidates[0], True


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return CONFUSABLE_COST if frozenset((a, b)) in CONFUSABLE else SUBSTITUTE_COST


def weighted_distance(a: str, b: str) -> float:
    """Levenshtein distance where confusable substitutions cost less.

    This is the function that lets the association engine in Milestone 7 accept
    an OCR near-miss without also accepting two different vehicles.
    """
    a, b = normalise(a), normalise(b)
    if not a or not b:
        return float(max(len(a), len(b)))

    prev = [j * GAP_COST for j in range(len(b) + 1)]
    for i, ca in enumerate(a, start=1):
        cur = [i * GAP_COST]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + GAP_COST,                       # deletion
                cur[j - 1] + GAP_COST,                    # insertion
                prev[j - 1] + substitution_cost(ca, cb),  # substitution
            ))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """0-1 similarity. 1.0 is identical; confusable swaps stay near 1."""
    a, b = normalise(a), normalise(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    longest = max(len(a), len(b))
    return max(0.0, 1.0 - weighted_distance(a, b) / longest)


def describe(text: str) -> str:
    """Human-readable verdict, used in logs and on the dashboard."""
    fmt = inspect(text)
    if not fmt.valid:
        return "does not match any Indian plate format"
    if fmt.kind == "bh":
        return "valid Bharat-series plate"
    return (f"valid {fmt.state} plate" if fmt.known_state
            else f"valid format, but {fmt.state!r} is not a known state code")
