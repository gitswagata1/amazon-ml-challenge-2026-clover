"""Text normalisation: Indic-script transliteration (stdlib unicodedata only), abbreviation expansion,
consonant skeletons (typo/transliteration tolerant), legal-suffix stripping, house-number keys."""
import os
import re
import unicodedata
from functools import lru_cache
from multiprocessing import Pool

import polars as pl

INDIC = {"DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM"}
IVOWELS = {"A", "AA", "I", "II", "U", "UU", "E", "EE", "AI", "O", "OO", "AU", "SHORT E", "SHORT O",
           "VOCALIC R", "VOCALIC RR", "VOCALIC L", "VOCALIC LL", "CANDRA E", "CANDRA O"}

ABBR = {
    "corp": "corporation", "co": "company", "inc": "incorporated", "lnc": "incorporated", "ltd": "limited",
    "pvt": "private", "pte": "private", "intl": "international", "mfg": "manufacturing", "svc": "services",
    "svcs": "services", "bros": "brothers", "assoc": "associates", "mgmt": "management", "natl": "national",
    "et": "and",
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway", "ste": "suite", "fl": "floor",
    "flr": "floor", "apt": "apartment", "nr": "near", "opp": "opposite", "pl": "place", "sq": "square",
    "ct": "court", "cir": "circle", "pkwy": "parkway", "bldg": "building", "rte": "route", "mkt": "market",
    "sec": "sector", "ngr": "nagar", "n": "north", "s": "south", "e": "east", "w": "west",
    "ets": "etablissements", "fbg": "faubourg", "chem": "chemin",
}
LEGAL = ("private limited incorporated corporation company llc llp plc lp the and of sa sarl sas sasu eurl "
         "gmbh pty opc pc pllc dba fka aka ms smt pa public elelpi elelsi ei cie snc selarl scp sca scop "
         "le la les de du des").split()  # last line: French articles, dropped from the core like 'the' / 'of'


def _vowel(v):
    return "ri" if v.startswith("VOCALIC") else v.split()[-1].lower()


@lru_cache(maxsize=None)
def _char(c):
    """-> (kind, latin). kind: c=consonant (inherent a), m=vowel sign, h=virama, z=ignore, x=literal."""
    if c.isascii():
        return "x", c
    try:
        name = unicodedata.name(c)
    except ValueError:
        return "x", " "
    script, _, rest = name.partition(" ")
    if script not in INDIC:
        d = "".join(ch for ch in unicodedata.normalize("NFKD", c) if not unicodedata.combining(ch))
        return "x", d
    if rest.startswith("LETTER "):
        v = rest[7:]
        if v in IVOWELS:
            return "x", _vowel(v)
        if v.startswith("CHILLU "):
            return "x", v[7:].lower()
        return "c", v.lower()[:-1] if v.endswith("A") and len(v) > 1 else v.lower()
    if rest.startswith("VOWEL SIGN "):
        return "m", _vowel(rest[11:])
    if rest in ("SIGN VIRAMA", "AU LENGTH MARK"):
        return "h", ""
    if rest in ("SIGN ANUSVARA", "SIGN CANDRABINDU"):
        return "x", "n"
    if rest == "SIGN VISARGA":
        return "x", "h"
    if rest.startswith("DIGIT "):
        return "x", str(unicodedata.digit(c))
    return "z", ""


def latin(s):
    """Transliterate Indic scripts to rough Latin and strip accents from everything else."""
    if s.isascii():
        return s
    out, pending = [], False
    for c in s:
        kind, lat = _char(c)
        if kind == "z":
            continue
        if pending and kind not in ("m", "h"):
            out.append("a")
        pending = kind == "c"
        out.append(lat)
    if pending:
        out.append("a")
    return "".join(out).replace("rrrr", "tt")  # Malayalam റ്റ


_LEET = str.maketrans("013568", "olesgb")  # digit-for-letter noise in the data: bui1ders, 6lobal, 8lue
_ORD = re.compile(r"\d+(st|nd|rd|th)")


def _tok(t):
    """Expand abbreviations; undo digit-for-letter typos inside words (freed0m, 5outhern); ordinals stay (1st, 3rd)."""
    if (not t.isalpha() and not t.isdigit() and sum(c.isalpha() for c in t) > len(t) // 2
            and not _ORD.fullmatch(t)):
        t = t.translate(_LEET)
    return ABBR.get(t, t)


_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\b)?")
DOTTED_LEGAL = {"sa", "sas", "sasu", "sarl", "eurl", "snc", "selarl", "scp", "sca", "scop",   # French
                "llc", "llp", "lp", "pc", "pllc", "pa", "plc"}                                   # US / generic


def _undot(m):
    """'s.a.s.' -> 'sas', 'l.l.c.' -> 'llc' (else 's a s' -> 'south a south'); only legal forms, other initials unchanged."""
    t = m.group().replace(".", "")
    return f" {t} " if t in DOTTED_LEGAL else m.group()


# 'St X' / 'Ste X' means Saint(e) after a French street type, at the start of a comma part or after a postcode
# ('Rue St Denis', ', St Louis', '75002 St Denis'); 'Main St, Ste 100' keeps street / suite.
# In names 'St X' / 'Ste X' is Saint(e) unless it follows a number ('Boulangerie St Michel', 'St. Jude'; '5th St Deli')
_SAINT = re.compile(r"(\b(?:rue|r|avenue|av|bd|boulevard|place|pl|quai|q|allee|all|chemin|ch|chem|impasse|imp|"
                    r"faubourg|fbg|cours|route|rte|square|passage|parvis)\.?\s+|^\s*|,\s*|\b\d{5}\s+)st(e?)\b\.?\s+(?=[a-z]{2})")
_NAME_SAINT = re.compile(r"(^\s*|(?<![0-9a-z])[a-z]+\s+)st(e?)\b\.?\s+(?=[a-z]{2})")
_ELISION = re.compile(r"\b(?:[ldj]|qu)['’](?=[a-z])")  # French l'/d'/j'/qu' ('L'Atelier', 'rue d'Alsace')
_BOXNUM = re.compile(r"\b(cedex|bp|cs|tsa)\s*\d+")  # French postal box / Cedex numbers are not house numbers


def norm(s, name=False):
    s = latin(s or "").lower().replace("&", " and ").replace("+", " and ")
    s = re.sub(r"^\s*m\s*/\s*s\b\.?", " ", s)  # Indian 'M/s Sharma Traders' prefix (else 'm south sharma')
    s = _DOTTED.sub(_undot, s)
    s = re.sub(r"\bn\s*°", " no ", s)  # French 'N° 6' (else 'north 6')
    s = re.sub(r"\bst(e?)\.?-(?=[a-z])", r"saint\1 ", s)  # 'St-Nazaire', 'Ste-Foy' (else 'street', 'suite')
    s = _SAINT.sub(r"\1saint\2 ", s) if not name else _NAME_SAINT.sub(r"\1saint\2 ", s)
    s = _BOXNUM.sub(r"\1", _ELISION.sub("", s))
    s = re.sub(r"(\d)(bis|ter|quater)\b", r"\1 \2", s)  # '59bis'
    s = re.sub(r"\.(com|c0m|in|net|org|co)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(map(_tok, s.split()))


_V = re.compile(r"[aeiouyh]")
_R = re.compile(r"(.)\1+")


_SK = str.maketrans("wbcqz", "vvkkj")


def skel(tok):
    """Consonant skeleton: marketing/मार्केटिंग -> mrktng, enterpires/enterprises -> ntrprs, tech/टेक -> tk."""
    return _R.sub(r"\1", _V.sub("", tok.replace("ph", "f").translate(_SK)))


LEGAL_W = set(LEGAL) | {"pvt", "ltd", "inc", "corp", "co"}
LEGAL_SK = {k for k in map(skel, LEGAL) if len(k) >= 3}  # short skeletons collide with real words (eurl/royal)


_NATIVE_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "native_dict.tsv")
# native-script / accented word -> English word, mined from training pairs by native_dict.py (empty if absent)
NATIVE = (dict(pl.read_csv(_NATIVE_TSV, separator="\t", quote_char=None).iter_rows())
          if os.path.exists(_NATIVE_TSV) else {})
_PUNCT = re.compile(r"[,()\[\]\-|&/]")


def translate(raw):
    """Replace mined native-script words by their English word ('குளோபல்' -> 'global'); ASCII names unchanged."""
    if not raw or raw.isascii() or not NATIVE:
        return raw
    return " ".join(NATIVE.get(t, t) for t in _PUNCT.sub(" ", raw.lower()).split())


def name_fields(raw):
    raw = translate(raw)
    n = norm(raw, name=True)
    toks = [(t, skel(t)) for t in n.split()]
    # legal words match exactly; the skeleton match is only for transliterated Indic text (praiveta -> prvt),
    # otherwise real words collide with legal skeletons (smith/summit ~ smt, israel ~ srl, place ~ plc)
    translit = not (raw or "").isascii() and any(_char(c)[0] in "cm" for c in raw)
    core = [(t, k) for t, k in toks if t not in LEGAL_W and not (translit and k in LEGAL_SK)] or toks
    return n, " ".join(t for t, _ in core), " ".join(k for _, k in core if k)


_NUM = re.compile(r"\d")
# generic street-type words / articles: skipped when picking the street word after a house number
# ('19 Rue de la Douve' -> '19 dv', '41 Ch Molière' -> '41 mlr')
STREET_SKIP = set("rue r route rte chemin ch allee impasse imp quai cours place avenue boulevard street road lane "
                  "drive way de la le les du des d l di del da the of number no unit suite floor apartment flat "
                  "plot house north south east west bis ter quater "
                  "all q fbg faubourg chem cedex bp cs saint sainte".split())


def addr_fields(raw):
    """Normalised address + house-number keys: '<digits> <next word skeleton>' within one comma part."""
    raw = latin(raw or "")
    keys = []
    for part in raw.split(","):
        toks = norm(part).split()
        for i, t in enumerate(toks):
            if _NUM.search(t):
                d = re.sub(r"\D", "", t).lstrip("0")
                nxt = next((skel(x) for x in toks[i + 1:] if not _NUM.search(x) and x not in STREET_SKIP and skel(x)), "")
                if not d or (len(t) < 2 and not nxt):
                    continue
                keys.append(f"{d} {nxt}" if nxt else d)
    return norm(raw), " | ".join(dict.fromkeys(keys))


def block_keys(sk, hn):
    """Name-based blocking keys: f=full sorted name, b=token pairs, p=prefix, t=long single token,
    x=name token + house number. Composite keys stay rare even for generic names."""
    toks = sorted(set(t for t in sk.split() if len(t) >= 2))
    keys = []
    if toks:
        keys.append("f:" + " ".join(toks))
    keys += [f"b:{a} {b}" for n, a in enumerate(toks[:6]) for b in toks[n + 1:6]]
    keys += [f"t:{t}" for t in toks if len(t) >= 5]
    flat = sk.replace(" ", "")
    if len(flat) >= 4:
        keys.append("p:" + flat[:6])
    nums = [h.split(" ")[0] for h in hn.split(" | ") if h][:3]
    keys += [f"x:{t} {n}" for t in sk.split()[:2] if len(t) >= 2 for n in nums]
    keys += [f"h:{h}" for h in hn.split(" | ") if len(h) >= 2]
    return " | ".join(dict.fromkeys(keys))


def _rows(args):
    name, addr = args
    n, core, sk = name_fields(name)
    a, hn = addr_fields(addr)
    return n, core, sk, a, hn, block_keys(sk, hn)


def prep(df: pl.DataFrame, workers=os.cpu_count()) -> pl.DataFrame:
    """Adds name, core, sk (core skeleton), addr, hn (house-number keys), bk (blocking keys); ' | '-joined."""
    with Pool(workers) as p:
        rows = p.map(_rows, zip(df["business_name"].to_list(), df["business_address"].to_list()), chunksize=20000)
    cols = list(zip(*rows)) if rows else [[]] * 6
    return df.with_columns(
        *(pl.Series(c, v, dtype=pl.String) for c, v in zip(("name", "core", "sk", "addr", "hn", "bk"), cols)),
        pl.col("country").str.strip_chars().str.to_lowercase().alias("ctry"),
    )


if __name__ == "__main__":
    if NATIVE:
        assert name_fields("குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்")[1] == name_fields("Global Business Pvt Ltd")[1], name_fields("குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்")
    assert skel(norm("मार्केटिंग")) == skel("marketing") == "mrktng", skel(norm("मार्केटिंग"))
    assert name_fields("राम मार्केटिंग प्राइवेट लिमिटेड")[2] == name_fields("Ram Marketing Pvt. Ltd.")[2]
    assert name_fields("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி")[2] == name_fields("Raj Investments LLP")[2]
    assert name_fields("valuetradingindustries.com")[1] == "valuetradingindustries"
    assert name_fields("Freed0m Polytechnic")[2] == name_fields("Freedom Polytechnic")[2]
    assert name_fields("विजन टेक लिमिटेड")[2] == name_fields("Vision Tech Limited")[2]  # exact via native_dict.tsv
    print(block_keys("sk bldrs", "12 mg | 7"))
    assert addr_fields("19 Rue de la Douve, Pornic")[1] == addr_fields("NO. 19 R. DE LA DOUVE, PORNIC")[1] == "19 dv"
    assert addr_fields("41 Chemin Molière, Dunkerque")[1] == addr_fields("0041 Ch Molière, Dunkerque")[1] == "41 mlr"
    assert addr_fields("3 Rue Docteur Jean Vincent, Bordeaux")[1] == "3 dktr"
    assert addr_fields("59 Bis Rue Gaston De Foix")[1] == addr_fields("59bis Rue Gaston De Foix")[1] == "59 gstn"
    assert addr_fields("N° 6 Rue Du Professeur Lavignolle")[0] == "no 6 rue du professeur lavignolle"
    assert addr_fields("04 Rue Robespierre, St-Nazaire")[0] == addr_fields("04 Rue Robespierre, Saint-Nazaire")[0]
    assert addr_fields("1 Rue X, St.-Herblain")[0] == addr_fields("1 Rue X, Saint Herblain")[0]
    assert addr_fields("5 Rue St-Jean")[1] == addr_fields("5 Rue Saint Jean")[1]
    assert addr_fields("12 Main St, Ste 100")[0] == "12 main street suite 100"  # US abbreviations unchanged
    assert addr_fields("221 Main St, Ste B")[0] == "221 main street suite b"
    assert addr_fields("8 All. des Tilleuls")[1] == addr_fields("8 Allée des Tilleuls")[1] == "8 tls"
    assert addr_fields("22 Fbg St Antoine")[1] == addr_fields("22 Faubourg Saint-Antoine")[1] == "22 ntn"
    assert addr_fields("BP 45, 69001 Lyon Cedex 01")[1] == addr_fields("69001 Lyon")[1] == "69001 ln"
    assert addr_fields("NC, Greensboro, Unit 301, 1502 Arctic Fox Circle")[1] == "301 | 1502 rktk"
    assert name_fields("Boulangerie St Michel")[1] == name_fields("Boulangerie Saint-Michel")[1]
    assert name_fields("Boucherie Ste Anne")[1] == "boucherie sainte anne"
    assert name_fields("Les Délices de Marie")[1:] == name_fields("Délices Marie")[1:]
    assert name_fields("L'Atelier")[1:] == name_fields("Atelier")[1:]
    assert name_fields("Ets Martin")[1:] == name_fields("Établissements Martin")[1:]
    assert name_fields("Bui1ders Supp1y")[2] == name_fields("Builders Supply")[2]
    assert name_fields("6lobal 8lue Traders")[2] == name_fields("Global Blue Traders")[2]
    assert name_fields("M/S Sharma Traders")[1] == name_fields("M/s. Sharma Traders")[1] == "sharma traders"
    assert addr_fields("12, 1st Cross, 3rd Main, Bangalore")[0] == "12 1st cross 3rd main bangalore"
    assert name_fields("Panier Industries S.A.S.")[1:] == name_fields("Panier Industries SAS")[1:]
    assert name_fields("Ama Centre S.A.R.L.")[1] == name_fields("Ama Centre SARL")[1] == "ama centre"
    assert name_fields("Smith LLC")[1] == name_fields("Smith L.L.C.")[1] == "smith"      # smith ~ smt collision
    assert name_fields("Summit Place Partners")[1] == "summit place partners"
    assert name_fields("Israel Trading SARL")[1] == "israel trading"
    assert name_fields("Samiti Services Pvt Ltd")[1] == "samiti services"
    assert name_fields("M.G. Traders L.L.C.")[0] == "m g traders llc"  # legal forms undotted, other initials unchanged
    for s in ["रॉयल सूर्य मैनेजमेंट", "ಸಿಲ್ವರ್ ಪ್ರೊಡಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "గ్రేట్ ఇంపెక్స్ లిమిటెడ్",
              "ബെസ്റ്റ് ഹോസ്പിറ്റാലിറ്റി പ്രൈവറ്റ് ലിമിറ്റഡ്", "Wilmington Créative Gálaxyedge"]:
        print(s, "->", name_fields(s))
    print("ok")
