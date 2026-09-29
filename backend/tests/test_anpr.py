"""Tests for plate grammar and temporal consensus.

No model, no video, no network -- this is pure logic, and it is where the
accuracy actually comes from.

    PYTHONPATH=backend python backend/tests/test_anpr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vision.consensus import PlateReading, build_consensus   # noqa: E402
from app.vision.plate_grammar import (inspect, is_valid, normalise,  # noqa: E402
                                      repair, similarity, weighted_distance)

_passed, _failed = 0, []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  pass  {name}")
    else:
        _failed.append(name)
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
print("\nnormalisation and validation")

check("strips spaces, dashes and case", normalise(" ka-01 ab 1234 ") == "KA01AB1234")
check("standard plate is valid", is_valid("KA01AB1234"))
check("single series letter is valid", is_valid("KA01A1234"))
check("three series letters are valid", is_valid("KA1ABC1234"))
check("Bharat series is valid", is_valid("21BH1234AB"))
check("random text is not a plate", not is_valid("HELLOWORLD"))
check("too few trailing digits is not a plate", not is_valid("KA01AB123"))
check("known state code recognised", inspect("MH12DE1433").known_state)
check("unknown state code flagged but format still valid",
      inspect("XX12DE1433").valid and not inspect("XX12DE1433").known_state)
check("unknown state scores lower than known",
      inspect("XX12DE1433").score < inspect("MH12DE1433").score)

# ---------------------------------------------------------------------------
print("\ngrammar repair")

for raw, want in [
    ("KA0IAB1234", "KA01AB1234"),   # I -> 1 in the district block
    ("KA01AB1Z34", "KA01AB1234"),   # Z -> 2 in the number block
    ("MH12OE1A33", "MH12OE1433"),   # A -> 4 in the number block
    ("TN09BX22S5", "TN09BX2255"),   # S -> 5
    ("KA0LAB12E4", "KA01AB1234"),   # two fixes in one read
]:
    got, changed = repair(raw)
    check(f"repairs {raw} -> {want}", got == want and changed, f"got {got}")

got, changed = repair("KA01AB1234")
check("never 'fixes' an already-valid plate", got == "KA01AB1234" and not changed)
check("refuses to guess at junk", repair("HELLO") == ("HELLO", False))
check("refuses wrong-length strings", repair("KA01AB12") == ("KA01AB12", False))

# ---------------------------------------------------------------------------
print("\nconfusion-aware similarity")

check("identical plates score 1.0", similarity("KA01AB1234", "KA01AB1234") == 1.0)
near = similarity("KA01AB1234", "KA0IAB1Z34")     # two confusable swaps
far = similarity("KA01AB1234", "KA01AB9999")      # four real differences
check("confusable near-miss stays high", near > 0.9, f"{near:.3f}")
check("genuinely different plates score much lower", far < 0.7, f"{far:.3f}")
check("confusable pair beats an unrelated pair by a wide margin",
      near - far > 0.2, f"{near:.3f} vs {far:.3f}")
check("a confusable swap costs less than an arbitrary one",
      weighted_distance("0", "O") < weighted_distance("0", "W"))
check("empty input scores 0", similarity("", "KA01AB1234") == 0.0)
check("similarity is symmetric",
      similarity("KA01AB1234", "KA0IAB1234") == similarity("KA0IAB1234", "KA01AB1234"))

# ---------------------------------------------------------------------------
print("\ntemporal consensus")

noisy = [
    PlateReading("KA01AB1234", 0.71), PlateReading("KA0IAB1234", 0.66),
    PlateReading("KA01AB1Z34", 0.58), PlateReading("KA01AB1234", 0.74),
    PlateReading("KAO1AB1234", 0.61), PlateReading("KA01AB1234", 0.69),
    PlateReading("KA01A81234", 0.55), PlateReading("KA01AB1234", 0.77),
    PlateReading("KA01AB1284", 0.52), PlateReading("KA01AB1234", 0.70),
]
c = build_consensus(noisy)
check("votes the right plate out of noisy reads", c.text == "KA01AB1234", c.text)
check("confidence beats every individual read",
      c.confidence > max(r.confidence for r in noisy), f"{c.confidence:.3f}")
check("reports how much evidence it used", c.reads_used == 10 and c.reads_total == 10)
check("knows the answer is a valid plate", c.format_valid)

garbage = build_consensus(noisy + [PlateReading("ZZ", 0.99)])
check("a short garbage read cannot hijack the vote", garbage.text == "KA01AB1234")
check("garbage read is excluded from the count", garbage.reads_used == 10,
      f"used={garbage.reads_used}")

single = build_consensus([PlateReading("KA01AB1234", 0.99)])
check("one frame is never enough to call a plate", not single.ok)

low = build_consensus([PlateReading("KA01AB1234", 0.05)] * 6)
check("reads the OCR itself doubts are dropped", not low.ok)

thin = build_consensus([PlateReading("KA01AB1234", 0.8)] * 2)
thick = build_consensus([PlateReading("KA01AB1234", 0.8)] * 12)
check("more agreeing evidence means more confidence",
      thick.confidence > thin.confidence, f"{thin.confidence:.3f} vs {thick.confidence:.3f}")

invalid = build_consensus([PlateReading("ZZZZZZZZZZ", 0.9)] * 6)
check("an unparseable plate is scored lower, not discarded",
      invalid.ok and not invalid.format_valid and invalid.confidence < thick.confidence)

# per-character confidences beat a whole-string average
per_char = [
    PlateReading("KA01AB1234", 0.5, char_confidence=[.9,.9,.9,.9,.9,.9,.9,.9,.9,.99]),
    PlateReading("KA01AB1234", 0.5, char_confidence=[.9,.9,.9,.9,.9,.9,.9,.9,.9,.99]),
    PlateReading("KA01AB1237", 0.9, char_confidence=[.9,.9,.9,.9,.9,.9,.9,.9,.9,.10]),
]
c2 = build_consensus(per_char)
check("a shaky character loses even when its whole read scored high",
      c2.text == "KA01AB1234", c2.text)

repaired = build_consensus([PlateReading("KA0IAB1Z34", 0.8)] * 5)
check("consensus applies grammar repair to the voted result",
      repaired.text == "KA01AB1234" and repaired.repaired, repaired.text)
check("keeps the pre-repair vote for the audit trail",
      repaired.raw_vote == "KA0IAB1Z34")
check("explain() is human readable", "reads" in repaired.explain())

check("no reads at all is handled", not build_consensus([]).ok)

# ---------------------------------------------------------------------------
print()
if _failed:
    print(f"{_passed} passed, {len(_failed)} FAILED: {', '.join(_failed)}")
    raise SystemExit(1)
print(f"{_passed} passed, 0 failed")
