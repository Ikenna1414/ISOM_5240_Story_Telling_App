"""Small unit tests for deterministic story helper functions."""

from app import clean_generated_text, count_words, limit_to_100_words


def test_clean_generated_text_removes_label_and_spacing():
    assert clean_generated_text("Story:  A fox  waved .") == "A fox waved."


def test_count_words():
    assert count_words("One small happy fox.") == 4


def test_limit_to_100_words():
    long_story = " ".join(f"word{i}" for i in range(105))
    shortened = limit_to_100_words(long_story)
    assert count_words(shortened) == 100
    assert shortened.endswith(".")
