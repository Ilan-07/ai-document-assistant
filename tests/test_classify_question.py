"""Tests for rag_pipeline.classify_question — regex-based question typing."""

from __future__ import annotations

import pytest

from rag_pipeline import FORMAT_RULES, classify_question


@pytest.mark.unit
@pytest.mark.parametrize(
    "question, expected",
    [
        # SUMMARY
        ("Summarize this document", "SUMMARY"),
        ("summarise the paper", "SUMMARY"),
        ("Give me an overview", "SUMMARY"),
        ("Describe this book", "SUMMARY"),
        # COMPARE
        ("Compare X and Y", "COMPARE"),
        ("X versus Y", "COMPARE"),
        ("What is the difference between A and B?", "COMPARE"),
        ("How do A and B differ?", "COMPARE"),
        # LIST
        ("List all employees", "LIST"),
        ("Name all the products", "LIST"),
        ("Enumerate the steps", "LIST"),
        ("What are all the SKUs?", "LIST"),
        # DEFINE
        ("Define throughput", "DEFINE"),
        ("Definition of latency", "DEFINE"),
        ("What does API mean?", "DEFINE"),
        # YES_NO
        ("Is the lab open?", "YES_NO"),
        ("Are tests required?", "YES_NO"),
        ("Can I work remotely?", "YES_NO"),
        ("Did the price change?", "YES_NO"),
        ("Should we ship today?", "YES_NO"),
        # WHO
        ("Who manages security?", "WHO"),
        # WHEN
        ("When was the policy updated?", "WHEN"),
        # WHERE
        ("Where is the office?", "WHERE"),
        # WHY
        ("Why did this fail?", "WHY"),
        # HOW (not "how many" / "how do" caught earlier)
        ("How does the system work?", "HOW"),
        # WHAT
        ("What is the policy?", "WHAT"),
        # DEFAULT fallthrough
        ("", "DEFAULT"),
        ("Random sentence without question word", "DEFAULT"),
        ("$$%%&&", "DEFAULT"),
    ],
)
def test_classify_question(question, expected):
    assert classify_question(question) == expected


@pytest.mark.unit
def test_format_rules_covers_all_labels():
    """Every label returned by classify_question must have a FORMAT_RULES entry."""
    labels = {
        "SUMMARY", "COMPARE", "LIST", "DEFINE", "YES_NO",
        "WHO", "WHEN", "WHERE", "WHY", "HOW", "WHAT", "DEFAULT",
    }
    assert labels.issubset(FORMAT_RULES.keys())


@pytest.mark.unit
def test_classify_question_specificity_ordering():
    """More-specific patterns must win over less-specific ones."""
    # "What are all the X" should match LIST, not WHAT
    assert classify_question("What are all the SKUs?") == "LIST"
    # "What does X mean" should match DEFINE, not WHAT
    assert classify_question("What does latency mean?") == "DEFINE"
    # "Compare X and Y" should match COMPARE even though it starts with a verb
    assert classify_question("Compare A and B") == "COMPARE"


@pytest.mark.unit
def test_classify_question_case_insensitive():
    assert classify_question("WHO is the CEO?") == "WHO"
    assert classify_question("list ALL items") == "LIST"


@pytest.mark.unit
def test_classify_question_leading_whitespace():
    assert classify_question("   when did this happen?") == "WHEN"
