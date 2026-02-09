import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MzansiLensResult:
    provinces: tuple[str, ...]
    places: tuple[str, ...]
    leaders: tuple[str, ...]
    issues: tuple[str, ...]
    parties: tuple[str, ...]
    institutions: tuple[str, ...]
    community_voices: tuple[str, ...]
    voice_quotes: tuple[str, ...]


def _dedupe_keep_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        v = (v or "").strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return tuple(out)


def _find_any(text: str, patterns: dict[str, str]) -> tuple[str, ...]:
    t = text or ""
    hits: list[str] = []
    for label, pat in patterns.items():
        if re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE):
            hits.append(label)
    return _dedupe_keep_order(hits)


_QUOTE_RE = re.compile(r"[\"\u201c\u201d]([^\"\u201c\u201d]{20,240})[\"\u201c\u201d]", flags=re.MULTILINE)
_ATTRIBUTION_RE = re.compile(
    r"\b(?:said|told|added|stated|warned|claimed|argued|according\s+to|spokesperson|spokesman|spokeswoman|union|ngo)\b",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _extract_voice_quotes(text: str, voice_patterns: dict[str, str], max_quotes: int = 3) -> tuple[str, ...]:
    t = text or ""
    if not t:
        return ()

    hits: list[str] = []
    window = 140

    for m in _QUOTE_RE.finditer(t):
        quote = (m.group(1) or "").strip()
        if not quote:
            continue

        head = t[max(0, m.start() - window) : m.start()]
        tail = t[m.end() : m.end() + window]
        ctx = (head + " " + tail).strip()

        # Only keep quotes that look attributed (reduces false positives).
        # Require an attribution cue near the quote boundaries.
        if not (_ATTRIBUTION_RE.search(tail[:80]) or _ATTRIBUTION_RE.search(head[-80:])):
            continue

        best_label = None
        best_dist = None

        for label, pat in voice_patterns.items():
            # Find closest actor mention to the quote (either before or after).
            for mm in re.finditer(pat, tail, flags=re.IGNORECASE | re.MULTILINE):
                dist = mm.start()
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_label = label
            for mm in re.finditer(pat, head, flags=re.IGNORECASE | re.MULTILINE):
                dist = (len(head) - mm.end())
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_label = label

        # Keep only reasonably local attributions.
        if best_label is None or (best_dist is not None and best_dist > 110):
            continue

        hits.append(f'{best_label}: "{quote}"')
        if len(hits) >= max(1, int(max_quotes)):
            break

    return _dedupe_keep_order(hits)


_PROVINCES = {
    "Eastern Cape": r"\bEastern\s+Cape\b",
    "Free State": r"\bFree\s+State\b",
    "Gauteng": r"\bGauteng\b",
    "KwaZulu-Natal": r"\bKwaZulu[-\s]?Natal\b|\bKZN\b",
    "Limpopo": r"\bLimpopo\b",
    "Mpumalanga": r"\bMpumalanga\b",
    "Northern Cape": r"\bNorthern\s+Cape\b",
    "North West": r"\bNorth\s+West\b",
    "Western Cape": r"\bWestern\s+Cape\b",
}

_PLACES = {
    "Johannesburg": r"\bJohannesburg\b|\bJoburg\b",
    "Pretoria": r"\bPretoria\b",
    "Tshwane": r"\bTshwane\b",
    "Cape Town": r"\bCape\s+Town\b",
    "Durban": r"\bDurban\b",
    "eThekwini": r"\beThekwini\b",
    "Gqeberha": r"\bGqeberha\b|\bPort\s+Elizabeth\b",
    "Bloemfontein": r"\bBloemfontein\b",
    "Polokwane": r"\bPolokwane\b",
    "Mbombela": r"\bMbombela\b|\bNelspruit\b",
    "Kimberley": r"\bKimberley\b",
    "Mahikeng": r"\bMahikeng\b|\bMafikeng\b",
}

_LEADERS = {
    "Cyril Ramaphosa": r"\bRamaphosa\b",
    "Jacob Zuma": r"\bJacob\s+Zuma\b|\bZuma\b",
    "Julius Malema": r"\bJulius\s+Malema\b|\bMalema\b",
    "John Steenhuisen": r"\bSteenhuisen\b",
    "Helen Zille": r"\bHelen\s+Zille\b|\bZille\b",
    "Paul Mashatile": r"\bMashatile\b",
    "Bheki Cele": r"\bBheki\s+Cele\b|\bCele\b",
    "Gwede Mantashe": r"\bMantashe\b",
}

_PARTIES = {
    "ANC": r"\bANC\b|\bAfrican\s+National\s+Congress\b",
    "DA": r"\bDA\b|\bDemocratic\s+Alliance\b",
    "EFF": r"\bEFF\b|\bEconomic\s+Freedom\s+Fighters\b",
    "MK": r"\bMK\b|\buMkhonto\s+we\s+Sizwe\b",
    "IFP": r"\bIFP\b|\bInkatha\s+Freedom\s+Party\b",
    "ActionSA": r"\bActionSA\b",
}

_INSTITUTIONS = {
    "Eskom": r"\bEskom\b",
    "Transnet": r"\bTransnet\b",
    "SABC": r"\bSABC\b|\bSouth\s+African\s+Broadcasting\s+Corporation\b",
    "PRASA": r"\bPRASA\b|\bPassenger\s+Rail\s+Agency\s+of\s+South\s+Africa\b",
    "SASSA": r"\bSASSA\b|\bSouth\s+African\s+Social\s+Security\s+Agency\b",
    "SARS": r"\bSARS\b|\bSouth\s+African\s+Revenue\s+Service\b",
 }

_ISSUES = {
    "Load-shedding": r"\bload\s*shedding\b|\bload-shedding\b",
    "Service delivery": r"\bservice\s+delivery\b",
    "Unemployment": r"\bunemployment\b|\bjobless\b",
    "Inflation": r"\binflation\b|\bcost\s+of\s+living\b",
    "Corruption": r"\bcorruption\b|\bstate\s+capture\b",
    "Crime": r"\bmurder\b|\brobbery\b|\bshooting\b|\bgang\b|\bkidnapp(?:ing|ed)\b",
    "GBV": r"\bGBV\b|\bgender[-\s]?based\s+violence\b",
    "Elections": r"\belection\b|\belections\b|\bIEC\b",
    "Electricity": r"\belectricity\b|\bpower\s+outage\b|\bblackout\b",
    "Water outages": r"\bwater\s+outage\b|\bwater\s+cuts\b|\bno\s+water\b",
    "Protests": r"\bprotest\b|\bprotests\b|\bdemonstration\b",
    "Education": r"\beducation\b|\bschools?\b|\bmatric\b|\buniversity\b",
    "Healthcare": r"\bhealth\s*care\b|\bhealthcare\b|\bhospital\b|\bclinic\b",
}


_COMMUNITY_VOICES = {
    # Unions
    "COSATU": r"\bCOSATU\b|\bCongress\s+of\s+South\s+African\s+Trade\s+Unions\b",
    "NUMSA": r"\bNUMSA\b|\bNational\s+Union\s+of\s+Metalworkers\s+of\s+South\s+Africa\b",
    "SADTU": r"\bSADTU\b|\bSouth\s+African\s+Democratic\s+Teachers\s+Union\b",
    "NEHAWU": r"\bNEHAWU\b|\bNational\s+Education\s+Health\s+and\s+Allied\s+Workers\s+Union\b",
    "SAMWU": r"\bSAMWU\b|\bSouth\s+African\s+Municipal\s+Workers\s+Union\b",
    "Solidarity": r"\bSolidarity\b",
    # NGOs / civil society
    "OUTA": r"\bOUTA\b|\bOrganisation\s+Undoing\s+Tax\s+Abuse\b",
    "Gift of the Givers": r"\bGift\s+of\s+the\s+Givers\b",
    "SECTION27": r"\bSECTION\s*27\b",
    "Equal Education": r"\bEqual\s+Education\b",
    "Right2Know": r"\bRight2Know\b|\bR2K\b",
    "Black Sash": r"\bBlack\s+Sash\b",
    "AfriForum": r"\bAfriForum\b",
    "Amnesty International": r"\bAmnesty\s+International\b",
    "Treatment Action Campaign": r"\bTreatment\s+Action\s+Campaign\b|\bTAC\b",
    "Doctors Without Borders": r"\bDoctors\s+Without\s+Borders\b|\bMSF\b|\bM\u00e9decins\s+Sans\s+Fronti\u00e8res\b",
    # Local leadership (lightweight signals)
    "Ward councillor": r"\bward\s+councillor\b",
    "Traditional leader": r"\btraditional\s+leader\b|\bchief\b|\bkgosi\b|\bamakhosi\b",
}


def analyze_mzansi_lens(text: str) -> MzansiLensResult:
    t = (text or "").strip()
    community_voices = _find_any(t, _COMMUNITY_VOICES)
    voice_quotes = _extract_voice_quotes(t, _COMMUNITY_VOICES)
    return MzansiLensResult(
        provinces=_find_any(t, _PROVINCES),
        places=_find_any(t, _PLACES),
        leaders=_find_any(t, _LEADERS),
        issues=_find_any(t, _ISSUES),
        parties=_find_any(t, _PARTIES),
        institutions=_find_any(t, _INSTITUTIONS),
        community_voices=community_voices,
        voice_quotes=voice_quotes,
    )


def format_mzansi_lens_markdown(result: MzansiLensResult) -> str:
    parts: list[str] = []

    def cap(values: tuple[str, ...], max_items: int = 10) -> tuple[tuple[str, ...], int]:
        if not values:
            return (), 0
        if len(values) <= max_items:
            return values, 0
        return values[:max_items], max(0, len(values) - max_items)

    def add(label: str, values: tuple[str, ...]):
        if values:
            kept, remaining = cap(values, 10)
            joined = ", ".join(kept)
            if remaining:
                joined = f"{joined}, … (+{remaining} more)"
            parts.append(f"**{label}:** {joined}")

    add("Provinces", result.provinces)
    add("Places", result.places)
    add("Leaders", result.leaders)
    add("Parties", result.parties)
    add("Institutions", result.institutions)
    add("Issues", result.issues)
    add("Community Voices", result.community_voices)

    if result.voice_quotes:
        kept, remaining = cap(result.voice_quotes, 2)
        lines = [f"- {q}" for q in kept]
        if remaining:
            lines.append(f"- \u2026 (+{remaining} more)")
        parts.append("**Quoted Voices (samples):**\n" + "\n".join(lines))

    if not parts:
        return "General (no strong SA-specific signals detected)."

    return "\n\n".join(parts)
