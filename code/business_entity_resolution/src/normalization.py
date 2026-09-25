"""
Deterministic text normalization for business names, addresses, and countries.

Design principles:
- Preserve original fields; always create new normalized columns.
- All transformations are deterministic and documented.
- No information is destroyed that could be useful downstream.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Legal suffix mapping (normalized form -> set of surface forms)
# Applied ONLY to the final token(s) of a business name.
# ---------------------------------------------------------------------------
LEGAL_SUFFIX_MAP = {
    "inc": {"inc", "inc.", "incorporated"},
    "corp": {"corp", "corp.", "corporation"},
    "llc": {"llc", "l.l.c.", "l.l.c"},
    "ltd": {"ltd", "ltd.", "limited", "[limited]"},
    "pvt": {"pvt", "pvt.", "private", "p."},
    "llp": {"llp", "l.l.p.", "l.l.p"},
    "co": {"co", "co."},
    "plc": {"plc", "plc.", "public limited company"},
    "lp": {"lp", "l.p.", "l.p"},
    "pllc": {"pllc"},
    "pa": {"pa", "p.a.", "p.a"},
    "pc": {"pc", "p.c.", "p.c"},
    "dba": {"dba", "d.b.a.", "d.b.a", "d/b/a"},
    "intl": {"intl", "intl.", "international"},
    "grp": {"grp", "grp.", "group"},
    "assoc": {"assoc", "assoc.", "associates", "association"},
    "svcs": {"svcs", "svcs.", "services"},
    "tech": {"tech", "tech.", "technologies", "technology"},
    "mgmt": {"mgmt", "mgmt.", "management"},
    "ent": {"ent", "ent.", "enterprises", "enterprise"},
    "mfg": {"mfg", "mfg.", "manufacturing"},
    "hldgs": {"hldgs", "hldgs.", "holdings"},
}

# Build reverse lookup: surface_form -> canonical
_SUFFIX_REVERSE = {}
for canonical, surface_forms in LEGAL_SUFFIX_MAP.items():
    for sf in surface_forms:
        _SUFFIX_REVERSE[sf] = canonical

# Address abbreviation mapping
ADDRESS_ABBREV = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "terrace": "ter",
    "nagar": "nagar",
    "marg": "marg",
}

# Precompile regex patterns
_RE_WHITESPACE = re.compile(r"\s+")
_RE_PUNCT_NOISE = re.compile(r"[*#\[\]\(\){}\"']")
_RE_MS_PREFIX = re.compile(r"^m/s\s+", re.IGNORECASE)
_RE_COMMA_SPACE = re.compile(r"\s*,\s*")
_RE_DIGIT_RUN = re.compile(r"\d+")


def _unicode_normalize(text: str) -> str:
    """NFKD normalize and strip diacritics where safe."""
    return unicodedata.normalize("NFKD", text)


def _base_clean(text: str) -> str:
    """Lowercase, unicode-normalize, collapse whitespace."""
    text = _unicode_normalize(text)
    text = text.lower().strip()
    text = _RE_WHITESPACE.sub(" ", text)
    return text


def normalize_name(name: str) -> str:
    """
    Normalize a business name.

    Steps:
    1. Unicode NFKD normalization
    2. Lowercase
    3. Remove noise characters: * # [ ] ( ) { } " '
    4. Remove M/s prefix
    5. Normalize & -> and
    6. Collapse whitespace
    7. Normalize known legal suffixes to canonical forms
    """
    if not name or not name.strip():
        return ""

    text = _base_clean(name)

    # Remove noise punctuation
    text = _RE_PUNCT_NOISE.sub("", text)

    # Remove M/s prefix (common Indian business prefix)
    text = _RE_MS_PREFIX.sub("", text)

    # Normalize ampersand
    text = text.replace("&", " and ")

    # Collapse whitespace again after substitutions
    text = _RE_WHITESPACE.sub(" ", text).strip()

    # Normalize legal suffixes (only at end of name)
    tokens = text.split()
    if tokens:
        # Check last 1-3 tokens for legal suffix patterns
        normalized_tokens = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            # Check multi-word suffixes (e.g., "private limited")
            if i < len(tokens) - 1:
                bigram = f"{tok} {tokens[i + 1]}"
                if bigram in _SUFFIX_REVERSE:
                    normalized_tokens.append(_SUFFIX_REVERSE[bigram])
                    i += 2
                    continue
            # Check single-word suffixes
            if tok in _SUFFIX_REVERSE:
                normalized_tokens.append(_SUFFIX_REVERSE[tok])
            else:
                # Remove trailing punctuation from token
                clean_tok = tok.rstrip(".,;:-")
                if clean_tok in _SUFFIX_REVERSE:
                    normalized_tokens.append(_SUFFIX_REVERSE[clean_tok])
                else:
                    normalized_tokens.append(tok)
            i += 1
        text = " ".join(normalized_tokens)

    return text


def name_tokens(normalized_name: str) -> set:
    """
    Extract informative tokens from a normalized business name.
    Filters out very short tokens and common stop words.
    """
    if not normalized_name:
        return set()

    stop_words = {
        "the", "of", "and", "for", "in", "at", "to", "a", "an",
        "inc", "corp", "llc", "ltd", "pvt", "llp", "co", "plc",
        "lp", "pa", "pc", "pllc", "dba",
    }

    tokens = normalized_name.split()
    return {t for t in tokens if len(t) >= 2 and t not in stop_words}


def normalize_address(address: str) -> str:
    """
    Normalize a business address.

    Steps:
    1. Unicode NFKD normalization
    2. Lowercase
    3. Remove noise punctuation
    4. Normalize common abbreviations
    5. Normalize commas
    6. Collapse whitespace
    """
    if not address or not address.strip():
        return ""

    text = _base_clean(address)

    # Remove noise characters
    text = _RE_PUNCT_NOISE.sub("", text)

    # Normalize commas to single comma+space
    text = _RE_COMMA_SPACE.sub(", ", text)

    # Normalize address abbreviations
    tokens = text.split()
    normalized = []
    for tok in tokens:
        clean = tok.rstrip(".,;:")
        if clean in ADDRESS_ABBREV:
            normalized.append(ADDRESS_ABBREV[clean])
        else:
            normalized.append(tok)
    text = " ".join(normalized)

    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def address_tokens(normalized_address: str) -> set:
    """Extract tokens from normalized address, including digit tokens.
    Strips trailing punctuation (commas, periods, etc.) from each token."""
    if not normalized_address:
        return set()
    result = set()
    for t in normalized_address.split():
        t = t.rstrip(".,;:-")
        if len(t) >= 2:
            result.add(t)
    return result


def extract_digits(text: str) -> str:
    """Extract all digit runs from text, concatenated with spaces."""
    if not text:
        return ""
    return " ".join(_RE_DIGIT_RUN.findall(text))


def normalize_country(country: str) -> str:
    """Normalize country string (simple lowercase + strip)."""
    if not country or not country.strip():
        return ""
    return country.strip().lower()


def char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from text (for similarity computation)."""
    if not text or len(text) < n:
        return set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}
