"""Language, script, sentence, and token primitives."""

from __future__ import annotations

import pytest

from app.db.models import Language
from app.nlp.language import detect_language, hinglish_marker_ratio
from app.nlp.script import Script, detect_script, profile_script
from app.nlp.sentences import split_sentences
from app.nlp.tokens import count_tokens

HINDI = "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं।"
ENGLISH = "India is a large country. Many languages are spoken here."
HINGLISH = "mujhe is document ka summary do aur batao ki isme kya likha hai"


class TestScript:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (HINDI, Script.DEVANAGARI),
            (ENGLISH, Script.LATIN),
            (HINGLISH, Script.LATIN),
            ("", Script.UNKNOWN),
            ("12345 !@#", Script.UNKNOWN),
        ],
    )
    def test_detects_script(self, text, expected):
        assert detect_script(text) == expected

    def test_hindi_with_english_acronyms_stays_devanagari(self):
        # Real circulars carry English clause IDs; that must not flip the script.
        text = "यह आदेश RTI अधिनियम के अंतर्गत जारी किया गया है।"
        assert detect_script(text) == Script.DEVANAGARI

    def test_heavy_code_mixing_is_mixed(self):
        assert (
            detect_script("यह एक महत्वपूर्ण document है जिसमें English content भी शामिल है")
            == Script.MIXED
        )

    def test_any_devanagari_is_never_labelled_english(self):
        """Below the mixed-script threshold the classifier calls this Latin, but
        the *language* must still not be English — that would send Hindi-bearing
        text down the English/Hinglish branch."""
        text = "यह एक document है with English content throughout"
        assert detect_script(text) == Script.LATIN
        assert detect_language(text).language != Language.EN
        assert detect_language(text).language != Language.HINGLISH

    def test_profile_reports_ratios(self):
        profile = profile_script(ENGLISH)
        assert profile.latin_ratio == 1.0
        assert profile.devanagari_ratio == 0.0


class TestSentences:
    def test_splits_on_devanagari_danda(self):
        sentences = split_sentences(HINDI)
        assert len(sentences) == 2
        assert sentences[0].text.endswith("।")

    def test_splits_on_double_danda(self):
        assert len(split_sentences("पहला वाक्य॥ दूसरा वाक्य॥")) == 2

    def test_does_not_split_abbreviations(self):
        sentences = split_sentences("Dr. Smith arrived. He was late.")
        assert len(sentences) == 2
        assert sentences[0].text == "Dr. Smith arrived."

    def test_does_not_split_decimals(self):
        assert len(split_sentences("The value is 3.5 units total.")) == 1

    def test_does_not_split_initials(self):
        assert len(split_sentences("R. K. Narayan wrote novels.")) == 1

    def test_offsets_map_back_to_source(self):
        text = "First one. Second one."
        for sentence in split_sentences(text):
            assert text[sentence.start : sentence.end] == sentence.text

    def test_paragraph_break_is_a_boundary(self):
        # Headings carry no terminator but must not merge into the next block.
        assert len(split_sentences("A heading\n\nSome body text follows here.")) == 2

    def test_empty_input(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []


class TestLanguage:
    def test_devanagari_is_hindi(self):
        assert detect_language(HINDI).language == Language.HI

    def test_english_is_english(self):
        assert detect_language(ENGLISH).language == Language.EN

    def test_romanized_hindi_is_hinglish(self):
        assert detect_language(HINGLISH).language == Language.HINGLISH

    def test_hinglish_suppressed_for_documents(self):
        # Documents are Hindi or English; a chunk must never be labelled Hinglish.
        assert detect_language(HINGLISH, allow_hinglish=False).language != Language.HINGLISH

    def test_plain_english_is_not_flagged_hinglish(self):
        # Guards against the English/Hinglish token collisions ("to", "me", "hi").
        for text in [
            "Please send the report to me by Friday.",
            "Hi there, this is his report and he wants it now.",
        ]:
            assert detect_language(text).language == Language.EN, text

    def test_marker_ratio_discriminates(self):
        assert hinglish_marker_ratio(HINGLISH) > hinglish_marker_ratio(ENGLISH)


class TestTokens:
    def test_devanagari_costs_more_tokens_per_char(self):
        """XLM-R tokenizes Devanagari far less efficiently than Latin. A naive
        chars/4 rule would undercount Hindi and produce oversized chunks."""
        hindi_short = HINDI[:50]
        english_short = ENGLISH[:50]
        assert count_tokens(hindi_short) > count_tokens(english_short)

    def test_empty_is_zero(self):
        assert count_tokens("") == 0

    def test_scales_with_length(self):
        assert count_tokens(ENGLISH * 4) > count_tokens(ENGLISH)
