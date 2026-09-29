"""Temporal plate consensus.

THE CORE IDEA OF MILESTONE 5, and the thing to say out loud to judges:
**we never trust a single frame.**

A vehicle is in view for 40-90 frames. Running OCR on each gives a spread of
readings -- most right, some off by a character, a few nonsense. Taking the
highest-confidence single read throws away everything the other frames knew.

Instead we vote, per character position, weighted by the OCR's own confidence.
Twelve frames saying "4" at position 8 with confidence 0.6 outweigh one frame
saying "A" at 0.9. Then the result goes through the plate grammar for a final
positional repair.

The output is not just a string. It carries per-character confidence, the
number of reads behind it, and whether it satisfies the plate format -- because
Milestone 7 needs to know *how much* to trust a plate, not merely what it says.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.vision.plate_grammar import inspect, normalise, repair


@dataclass
class PlateReading:
    """One OCR attempt on one frame."""

    text: str
    confidence: float
    frame_index: int = 0
    plate_width_px: int = 0
    # fast-alpr gives a confidence PER CHARACTER. When we have it, each
    # position votes with its own weight instead of the whole-string average --
    # so one shaky character does not drag down the eight the model was sure of.
    char_confidence: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = normalise(self.text)

    def weight_at(self, index: int) -> float:
        """Per-character weight, but ONLY when the array really lines up.

        fast-plate-ocr returns a FIXED-LENGTH confidence array -- one entry per
        model slot (9 or 10), padding included -- while ``text`` is the trimmed
        result. Indexing the array by text position therefore paired character
        0 with the confidence of whatever slot happened to be first, which was
        usually padding. The weights were silently wrong for every read whose
        text was shorter than the slot count, which is most of them.

        If the lengths do not match we cannot know the mapping, so we fall back
        to the whole-string confidence rather than using a misaligned number.
        """
        if len(self.char_confidence) == len(self.text) and index < len(self.char_confidence):
            return self.char_confidence[index]
        return self.confidence

    @property
    def quality(self) -> float:
        """Evidence weight for this read.

        A read taken off a 190px plate is worth more than one off a 90px plate,
        and we have the measurements to prove it. Used when reads disagree
        about how many characters the plate even has.
        """
        return self.confidence * (1.0 + min(1.0, self.plate_width_px / 200.0))


@dataclass
class PlateConsensus:
    """The agreed plate for one track, with its supporting evidence."""

    text: str = ""
    confidence: float = 0.0
    char_confidence: list[float] = field(default_factory=list)
    reads_used: int = 0
    reads_total: int = 0
    agreement: float = 0.0          # how much the reads agreed with each other
    format_valid: bool = False
    repaired: bool = False
    raw_vote: str = ""              # what voting produced, before repair
    status: str = "unreadable"      # confirmed | tentative | unreadable

    @property
    def ok(self) -> bool:
        """True only when we are willing to put this plate on a screen.

        Deliberately stricter than "we produced some characters". A consensus
        we do not trust reports no plate at all, because a wrong plate shown
        as fact is worse than an honest blank.
        """
        return bool(self.text) and self.status in ("confirmed", "tentative")

    def explain(self) -> str:
        """One line a judge can read off the dashboard."""
        if not self.text:
            return "no readable plate"
        bits = [self.status,
                f"{self.reads_used}/{self.reads_total} reads",
                f"agreement {self.agreement:.0%}",
                "format valid" if self.format_valid else "format invalid"]
        if self.repaired:
            bits.append(f"repaired from {self.raw_vote}")
        return "  |  ".join(bits)


def _weighted_mode_length(readings: list[PlateReading]) -> int:
    """The plate length the reads collectively believe in.

    Weighted by ``quality``, not raw confidence, and that change matters.
    A truncated read off a small crop comes back at 1.00 confidence, so under
    the old plain-confidence vote the truncated length simply won on numbers --
    small-vehicle frames outnumber close-up frames. The correct full-length
    reads were then dropped before voting even started, which is precisely how
    a confident wrong plate reached the dashboard.
    """
    votes: dict[int, float] = defaultdict(float)
    for r in readings:
        votes[len(r.text)] += r.quality
    return max(votes.items(), key=lambda kv: kv[1])[0]


def build_consensus(
    readings: list[PlateReading],
    min_confidence: float = 0.30,
    min_reads: int = 2,
    target_reads: int = 8,
    confirm_confidence: float = 0.60,
    tentative_confidence: float = 0.40,
) -> PlateConsensus:
    """Fold many noisy reads of one vehicle into one answer with a confidence.

    ``target_reads`` is the number of reads at which we stop discounting for
    thin evidence -- a plate seen twice should not score like one seen twenty
    times, even if both agree perfectly.
    """
    total = len(readings)
    usable = [r for r in readings if r.text and r.confidence >= min_confidence]
    if len(usable) < min_reads:
        return PlateConsensus(reads_total=total, reads_used=len(usable))

    # Only reads that agree on length can vote position-by-position. A read of
    # a different length is usually a partial or a merged detection.
    length = _weighted_mode_length(usable)
    aligned = [r for r in usable if len(r.text) == length]
    if len(aligned) < min_reads:
        return PlateConsensus(reads_total=total, reads_used=len(aligned))

    # Per-position, confidence-weighted vote.
    chars: list[str] = []
    char_conf: list[float] = []
    for i in range(length):
        votes: dict[str, float] = defaultdict(float)
        for r in aligned:
            votes[r.text[i]] += r.weight_at(i)
        best_char, best_weight = max(votes.items(), key=lambda kv: kv[1])
        chars.append(best_char)
        char_conf.append(best_weight / sum(votes.values()))

    raw = "".join(chars)
    text, was_repaired = repair(raw)
    fmt = inspect(text)

    # Overall confidence combines four independent things:
    #   how strongly each position was agreed on,
    #   how good the underlying reads were,
    #   how much evidence there was at all,
    #   whether the answer is a plausible plate.
    agreement = sum(char_conf) / len(char_conf)
    evidence = min(1.0, len(aligned) / target_reads)
    plausibility = 0.55 + 0.45 * fmt.score        # never zero it out entirely

    # READ QUALITY. Without this term the score measures only whether the reads
    # agreed with EACH OTHER -- the per-character weights cancel in the ratio
    # above, so ten reads at 0.05 that happen to agree scored exactly like ten
    # at 0.99. Three mutually-disagreeing reads the OCR itself scored ~0.41
    # came out "confirmed" at 0.67, which is precisely the kind of false
    # certainty this module exists to prevent. Agreement between doubtful reads
    # is not the same thing as confidence.
    #
    # DAMPED, NOT CAPPED, and the difference matters. Multiplying by raw read
    # quality would cap the consensus at the mean of its own reads, destroying
    # the property this whole module exists for: twelve agreeing reads SHOULD
    # be worth more than any one of them alone. The 0.5 floor keeps evidence
    # accumulating while still making poor reads cost something.
    read_quality = sum(r.confidence for r in aligned) / len(aligned)

    confidence = round(min(1.0, agreement * (0.5 + 0.5 * read_quality)
                           * (0.55 + 0.45 * evidence) * plausibility), 4)

    # Three states, not two. "We produced characters" and "we believe them" are
    # different claims, and the operator is entitled to know which one this is.
    if confidence >= confirm_confidence and fmt.valid:
        status = "confirmed"
    elif confidence >= tentative_confidence:
        status = "tentative"
    else:
        status = "unreadable"

    return PlateConsensus(
        text=text,
        confidence=confidence,
        char_confidence=[round(c, 3) for c in char_conf],
        reads_used=len(aligned),
        reads_total=total,
        agreement=round(agreement, 4),
        format_valid=fmt.valid,
        repaired=was_repaired,
        raw_vote=raw,
        status=status,
    )
