"""Unit tests for the competition pipeline's text normalisation, blocking keys and feature helpers.

Each case pins a behaviour the pipeline depends on, most of them taken from the docstrings and comments in
code/business_entity_resolution/src. Run from the repository root:  pytest -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))

import features  # noqa: E402
from prep import addr_fields, block_keys, latin, name_fields, norm, skel  # noqa: E402


# ── transliteration and normalisation ────────────────────────────────────────
def test_latin_strips_accents_and_leaves_ascii_alone():
    assert latin("Café") == "Cafe"
    assert latin("Acme Traders") == "Acme Traders"


@pytest.mark.parametrize("raw, expected", [
    ("M/s Sharma Traders", "sharma traders"),            # Indian 'M/s' prefix dropped
    ("bui1ders 6lobal 1st", "builders global 1st"),      # digit-for-letter typos fixed, ordinals kept
    ("S.A.S. Boulangerie", "sas boulangerie"),           # dotted legal form collapsed
    ("L'Atelier rue d'Alsace", "atelier rue alsace"),    # French elision removed
    ("St-Nazaire", "saint nazaire"),                     # hyphenated St = Saint
    ("Rue St Denis, 75002 St Denis", "rue saint denis 75002 saint denis"),
    ("Main St, Ste 100", "main street suite 100"),       # US street / suite stay street / suite
    ("N° 6 rue X", "no 6 rue x"),
    ("59bis rue", "59 bis rue"),
    ("acme.com", "acme"),
])
def test_norm_addresses(raw, expected):
    assert norm(raw) == expected


def test_norm_names_treat_st_as_saint_unless_after_a_number():
    assert norm("Boulangerie St Michel", name=True) == "boulangerie saint michel"
    assert norm("5th St Deli", name=True) == "5th street deli"


# ── consonant skeletons: the same word in English and Devanagari meets on one key ──
@pytest.mark.parametrize("a, b, sk", [
    ("marketing", "मार्केटिंग", "mrktng"),
    ("enterpires", "enterprises", "ntrprs"),   # typo and correct spelling collide, as intended
    ("tech", "टेक", "tk"),
])
def test_skeleton_matches_across_scripts_and_typos(a, b, sk):
    assert skel(norm(a)) == skel(norm(b)) == sk


# ── name and address fields ─────────────────────────────────────────────────
def test_name_fields_drop_legal_words_from_the_core():
    full, core, sk = name_fields("Sharma Traders Pvt Ltd")
    assert full == "sharma traders private limited"
    assert core == "sharma traders"
    assert sk == "srm trdrs"


def test_name_fields_keep_real_words_that_look_like_legal_skeletons():
    # 'smith' and 'summit' skeletonise to 'smt' (like the legal form 'smt'); for ASCII names they must stay
    _, core, _ = name_fields("Smith & Summit LLC")
    assert core == "smith summit"


def test_addr_fields_house_number_keys_skip_street_types():
    assert addr_fields("19 Rue de la Douve, 75002 Paris")[1] == "19 dv | 75002 prs"
    assert addr_fields("12345 Main St, Austin TX 78701")[1] == "12345 mn | 78701"


# ── blocking keys ───────────────────────────────────────────────────────────
def test_block_keys_cover_every_key_family():
    keys = block_keys("shrm trdrs", "12 mn").split(" | ")
    assert keys == ["f:shrm trdrs", "b:shrm trdrs", "t:trdrs", "p:shrmtr",
                    "x:shrm 12", "x:trdrs 12", "h:12 mn"]
    assert len(keys) == len(set(keys))  # no duplicates


def test_block_keys_empty_name_gives_no_name_keys():
    assert block_keys("", "") == ""


# ── feature helpers ─────────────────────────────────────────────────────────
def test_postcode_is_the_last_long_number_not_the_house_number():
    assert features._post("12345 main st austin tx 78701") == "78701"
    assert features._post("no digits here") == ""


def test_jaccard_and_missing_value_marker():
    out = features._jacc(["a b", "a", ""], ["b c", "a", "x"], " ")
    assert out[0] == pytest.approx(1 / 3)
    assert out[1] == 1.0
    assert out[2] == -1  # one side empty -> -1, so the model can tell 'missing' from 'no overlap'
