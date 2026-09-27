"""Organization names: normalization, classification and candidate extraction.

Source records name organizations in many ways: ``Zentavia Pharma``, ``Zentavia Pharma, Inc.``,
``ZENTAVIA PHARMA INC`` and ``Zentavia Pharmaceuticals Ltd`` are one company, while
``Orvexa Bio`` and ``Orvexa Labs`` are two. This module turns a raw name into a set of
**match keys** that resolve those differences deterministically, decides what kind of
organization it is, and pulls organization names out of publication affiliation strings.

The keys, in the order they are tried:

1. ``exact`` - the original name, lower-cased;
2. ``normalized`` - punctuation removed, whitespace collapsed, legal suffixes stripped;
3. ``expanded`` - normalized, with common abbreviations spelled out (``Pharma`` ->
   ``pharmaceuticals``, ``Tx`` -> ``therapeutics``, ``Labs`` -> ``laboratories``);
4. ``compact`` / ``expanded_compact`` - the same without spaces, so ``OrvexaBio`` matches
   ``Orvexa Bio``.

Each step keeps the words that distinguish one organization from another, so a match through
any key is safe. Fuzzy matching and the decision to merge live in
:mod:`cews.normalization.resolver`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from cews.constants import MAX_ORG_NAME_LENGTH, OrganizationType

# Legal forms that do not distinguish one organization from another.
LEGAL_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "llc",
        "lp",
        "llp",
        "plc",
        "corp",
        "corporation",
        "co",
        "company",
        "gmbh",
        "mbh",
        "ag",
        "kgaa",
        "kg",
        "sa",
        "sas",
        "sarl",
        "spa",
        "srl",
        "nv",
        "bv",
        "ab",
        "as",
        "asa",
        "oy",
        "oyj",
        "aps",
        "kk",
        "pte",
        "pty",
        "pvt",
        "sdn",
        "bhd",
        "cv",
        "ou",
        "doo",
        "dd",
        "sro",
        "jsc",
        "pjsc",
        "ulc",
        "holdings",
        "holding",
    }
)
# Abbreviations expanded so short and long spellings meet in the middle.
ABBREVIATIONS: dict[str, str] = {
    "pharma": "pharmaceuticals",
    "pharms": "pharmaceuticals",
    "pharm": "pharmaceuticals",
    "pharmaceutical": "pharmaceuticals",
    "tx": "therapeutics",
    "thera": "therapeutics",
    "ther": "therapeutics",
    "therapeutic": "therapeutics",
    "bio": "biosciences",
    "biosci": "biosciences",
    "bioscience": "biosciences",
    "biotech": "biotechnology",
    "lab": "laboratories",
    "labs": "laboratories",
    "laboratory": "laboratories",
    "gen": "genetics",
    "genetic": "genetics",
    "sci": "sciences",
    "science": "sciences",
    "tech": "technologies",
    "technology": "technologies",
    "med": "medical",
    "medicine": "medical",
    "res": "research",
    "inst": "institute",
    "intl": "international",
    "natl": "national",
    "univ": "university",
    "dept": "department",
    "ctr": "center",
    "centre": "center",
    "hosp": "hospital",
    "st": "saint",
    "svcs": "services",
    "svc": "services",
}
# Words that mark a division or a separate line of business; never ignored when matching.
DIVISION_WORDS: frozenset[str] = frozenset(
    {
        "oncology",
        "neurology",
        "immunology",
        "genetics",
        "diagnostics",
        "devices",
        "vaccines",
        "laboratories",
        "biosciences",
        "biotechnology",
        "ventures",
        "capital",
        "consumer",
        "animal",
        "veterinary",
        "digital",
        "services",
        "solutions",
        "manufacturing",
        "research",
        "development",
        "international",
        "global",
        "americas",
        "europe",
        "asia",
        "japan",
        "china",
        "india",
        "canada",
        "australia",
        "sciences",
        "technologies",
        "therapeutics",
        "pharmaceuticals",
        "medical",
        "health",
        "healthcare",
        "group",
    }
)

UNIVERSITY_WORDS = (
    "university",
    "univ",  # "Univ. of Harrowgate"
    "universite",
    "universitat",
    "universidad",
    "universita",
    "universiteit",
    "college",
    "polytechnic",
    "hochschule",
    "ecole",
    "academy",
    "school of",
    "faculty of",
)
HOSPITAL_WORDS = (
    "hospital",
    "hopital",
    "krankenhaus",
    "clinic",
    "klinik",
    "clinica",
    "medical center",
    "medical centre",
    "medical ctr",
    "health system",
    "infirmary",
    "cancer center",
    "cancer centre",
    "policlinico",
)
GOVERNMENT_WORDS = (
    "ministry",
    "research council",
    "council of",
    "national institute",
    "national institutes",
    "federal",
    "government",
    "authority",
    "bureau",
    "commission",
    "public health",
    "administration",
    "agency",
)
NONPROFIT_WORDS = ("foundation", "charity", "charitable", "association", "society", "nonprofit")
COMPANY_WORDS = (
    "pharmaceutical",
    "pharma",
    "therapeutics",
    "biosciences",
    "bioscience",
    "biotech",
    "laboratories",
    "labs",
    "diagnostics",
    "medicines",
    "medtech",
    "genomics",
    "genetics",
    "biologics",
    "biopharma",
)
RESEARCH_WORDS = ("institute", "research institute", "research center", "research centre")
# Company names are often shortened to a trailing abbreviation. These are only recognised as a
# whole final word, so "Bioethics Institute" is not mistaken for a company.
COMPANY_ABBREVIATIONS = frozenset({"bio", "tx", "gen", "dx", "rx", "sci", "labs", "pharma"})

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]")
_ADDRESS_HINT = re.compile(
    r"(?i)\b(department|dept|division|school|faculty|institute of|p\.?o\.? box|street|str\.|"
    r"stra(?:ss|ß)e|avenue|road|building|floor|suite|campus|\d{4,}|[a-z]{1,2}\d{1,2} ?\d[a-z]{2})\b"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_COUNTRY_OR_STATE = re.compile(
    r"(?i)^(usa|u\.s\.a\.?|united states|uk|u\.k\.?|united kingdom|england|scotland|wales|"
    r"germany|deutschland|france|italy|spain|japan|china|india|canada|australia|switzerland|"
    r"netherlands|belgium|sweden|denmark|norway|finland|ireland|israel|korea|singapore|brazil|"
    r"mexico|poland|austria|portugal|greece|turkey|russia|[a-z]{2})$"
)


@dataclass(frozen=True)
class NormalizedName:
    """A raw organization name together with the keys used to match it."""

    original: str
    normalized: str
    expanded: str
    compact: str
    expanded_compact: str
    tokens: tuple[str, ...]
    expanded_tokens: tuple[str, ...]
    suffixes: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        """True when nothing usable is left after normalization."""
        return not self.normalized

    def match_keys(self) -> tuple[tuple[str, str], ...]:
        """``(method, key)`` pairs in the order they should be tried."""
        pairs = (
            ("exact", self.original.casefold()),
            ("normalized", self.normalized),
            ("expanded", self.expanded),
            ("compact", self.compact),
            ("expanded_compact", self.expanded_compact),
        )
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for method, key in pairs:
            if key and key not in seen:
                seen.add(key)
                unique.append((method, key))
        return tuple(unique)


def normalize_organization_name(name: str) -> NormalizedName:
    """Return ``name`` with its match keys.

    Accents are folded, ``&`` becomes ``and``, punctuation is dropped, whitespace is collapsed
    and trailing legal suffixes are removed. The original spelling is always preserved.

    Raises:
        ValueError: if ``name`` is not a string or is longer than ``MAX_ORG_NAME_LENGTH``.
    """
    if not isinstance(name, str):
        raise ValueError(f"organization name must be a string, got {type(name).__name__}")
    original = " ".join(name.split())
    if len(original) > MAX_ORG_NAME_LENGTH:
        raise ValueError(f"organization name longer than {MAX_ORG_NAME_LENGTH} characters")

    folded = unicodedata.normalize("NFKD", original.casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    cleaned = _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", folded.replace("&", " and "))).strip()

    tokens = cleaned.split()
    suffixes: list[str] = []
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        suffixes.insert(0, tokens.pop())
    while len(tokens) > 1 and tokens[-1] == "and":
        tokens.pop()
    if len(tokens) > 1 and tokens[0] == "the":
        tokens.pop(0)

    expanded = [ABBREVIATIONS.get(token, token) for token in tokens]
    normalized_text = " ".join(tokens)
    expanded_text = " ".join(expanded)
    return NormalizedName(
        original=original,
        normalized=normalized_text,
        expanded=expanded_text,
        compact=_NON_ALNUM.sub("", normalized_text),
        expanded_compact=_NON_ALNUM.sub("", expanded_text),
        tokens=tuple(tokens),
        expanded_tokens=tuple(expanded),
        suffixes=tuple(suffixes),
    )


_ABBREVIATION = re.compile(r"\b[A-Za-z]{1,4}\.")


def display_quality(text: str) -> tuple[int, int, int, int]:
    """How presentable a spelling is, for choosing what to show on a dashboard.

    Several spellings of one company reduce to the same matching key ("Zentavia Pharma",
    "Zentavia Pharma, Inc." and "ZENTAVIA PHARMA INC" all match), so the stored display name
    would otherwise be whichever arrived first. Higher is better, in order: mixed case beats
    SHOUTING and all-lowercase; a name without an abbreviated word ("Nexoria Gen.") beats one
    with; the plain business name beats the legal form ("Zentavia Pharma" over "Zentavia Pharma,
    Inc."); and a longer name breaks the remaining ties.
    """
    letters = [character for character in text if character.isalpha()]
    mixed = int(bool(letters) and not text.isupper() and not text.islower())
    spelled_out = int(_ABBREVIATION.search(text) is None)
    tokens = [token.strip(".,") for token in text.casefold().split()]
    plain = int(not tokens or tokens[-1] not in LEGAL_SUFFIXES)
    return (mixed, spelled_out, plain, len(text))


@dataclass(frozen=True)
class OrganizationClass:
    """What kind of organization a name looks like, and why."""

    organization_type: OrganizationType
    confidence: float
    reason: str


def _contains(text: str, words: tuple[str, ...]) -> str | None:
    for word in words:
        cleaned = word.strip()
        if cleaned and re.search(rf"\b{re.escape(cleaned)}", text):
            return cleaned
    return None


def classify_organization_type(name: str) -> OrganizationClass:
    """Classify a name as a company, university, hospital, government body or non-profit.

    Competitor discovery keeps only companies by default, so this decides what may become a
    monitored competitor. Matching is keyword-based and reports the keyword it used.
    """
    text = " ".join(name.casefold().split())
    for words, kind, confidence in (
        (UNIVERSITY_WORDS, OrganizationType.UNIVERSITY, 0.9),
        (HOSPITAL_WORDS, OrganizationType.HOSPITAL, 0.9),
        (GOVERNMENT_WORDS, OrganizationType.GOVERNMENT, 0.85),
        (NONPROFIT_WORDS, OrganizationType.NONPROFIT, 0.8),
    ):
        found = _contains(text, words)
        if found:
            return OrganizationClass(kind, confidence, f"matched {found!r}")
    found = _contains(text, COMPANY_WORDS)
    if found:
        return OrganizationClass(OrganizationType.COMPANY, 0.8, f"matched {found!r}")
    final_word = text.split()[-1].strip(".") if text.split() else ""
    if len(text.split()) > 1 and final_word in COMPANY_ABBREVIATIONS:
        return OrganizationClass(
            OrganizationType.COMPANY, 0.6, f"shortened company name {final_word!r}"
        )
    found = _contains(text, RESEARCH_WORDS)
    if found:
        # "Institute" alone is ambiguous (academic, non-profit or corporate R&D).
        return OrganizationClass(OrganizationType.NONPROFIT, 0.5, f"matched {found!r}")

    normalized = normalize_organization_name(name)
    if normalized.suffixes:
        return OrganizationClass(
            OrganizationType.COMPANY, 0.75, f"legal suffix {normalized.suffixes[-1]!r}"
        )
    return OrganizationClass(OrganizationType.UNKNOWN, 0.3, "no identifying words")


def _looks_like_organization(text: str) -> bool:
    lowered = " ".join(text.casefold().split())
    groups = (
        COMPANY_WORDS,
        UNIVERSITY_WORDS,
        HOSPITAL_WORDS,
        NONPROFIT_WORDS,
        GOVERNMENT_WORDS,
        RESEARCH_WORDS,
        ("center", "centre", "group", "consortium", "network"),
    )
    if any(_contains(lowered, words) for words in groups):
        return True
    return bool(normalize_organization_name(text).suffixes)


def split_affiliation(text: str) -> list[str]:
    """Pull candidate organization names out of an affiliation string.

    ``"Fixture Oncology, Inc., Boston, MA, USA."`` gives ``["Fixture Oncology, Inc."]``: address
    parts, departments and e-mail addresses are dropped, and a legal suffix is joined back onto
    the name it belongs to. Affiliations are messy free text, so this is a best effort and
    callers store the result with a low relationship confidence.
    """
    if not text or not text.strip():
        return []
    candidates: list[str] = []
    for chunk in re.split(r"[;|]", _EMAIL.sub(" ", text)):
        merged: list[str] = []
        for raw_part in chunk.split(","):
            part = raw_part.strip(" .")
            if not part:
                continue
            bare = _PUNCTUATION.sub("", part).strip().casefold()
            if merged and bare in LEGAL_SUFFIXES:
                merged[-1] = f"{merged[-1]}, {part}"
                continue
            merged.append(part)
        for part in merged:
            if _COUNTRY_OR_STATE.match(part) or _ADDRESS_HINT.search(part):
                continue
            if len(part) < 3 or not re.search(r"[A-Za-z]{3}", part):
                continue
            candidates.append(part)
    if not candidates:
        return []
    named = [c for c in candidates if _looks_like_organization(c)]
    return named[:3] if named else candidates[:1]


def shares_distinguishing_words(left: NormalizedName, right: NormalizedName) -> bool:
    """True when two names differ only by words that do not identify a different organization.

    ``Orvexa Biosciences`` and ``Orvexa Laboratories`` differ by a division word, so they are
    **not** interchangeable; ``Zentavia Pharmaceuticals`` and ``Zentavia Pharmaceuticals Inc``
    are.
    """
    difference = set(left.expanded_tokens).symmetric_difference(right.expanded_tokens)
    return not (difference & DIVISION_WORDS)
