"""The protocol tests the researcher's own question; literature stays on topic."""

from __future__ import annotations

from cyberlab_literature import Paper
from cyberlab_orchestrator.agents.literature import relevance
from cyberlab_orchestrator.agents.methodology import _enforce_intent
from cyberlab_protocol import default_protocol
from cyberlab_protocol.intent import research_intent

GOAL = (
    "Investigate how well logistic regression, random forest, XGBoost and an MLP detect the "
    "rarest attack classes. Use balanced accuracy as the primary metric and report per-class "
    "recall. Test whether random forest achieves higher balanced accuracy than logistic "
    "regression, and whether XGBoost achieves higher balanced accuracy than the MLP."
)


def test_stated_hypotheses_and_primary_metric_are_read_from_the_goal() -> None:
    intent = research_intent(GOAL, [])
    assert intent["primary_metric"] == "balanced_accuracy"
    assert intent["hypotheses"] == [
        ("random_forest", "logistic_regression", "balanced_accuracy"),
        ("xgboost", "mlp", "balanced_accuracy"),
    ]
    assert (
        research_intent("Compare four models on CIC-IDS2017 in general terms.", [])["hypotheses"]
        == []
    )


def test_protocol_is_forced_to_test_the_goal_and_drops_untestable_questions() -> None:
    base = default_protocol("p", "cicids2017-mlcsv-v1")
    base = base.model_copy(
        update={
            "research_questions": [
                *base.research_questions,
                "What is the real-time performance and scalability of these models?",
            ]
        }
    )
    proto, notes = _enforce_intent(base, research_intent(GOAL, []), GOAL)
    assert [(h.comparison, h.primary_metric) for h in proto.hypotheses] == [
        ("random_forest > logistic_regression", "balanced_accuracy"),
        ("xgboost > mlp", "balanced_accuracy"),
    ]
    assert proto.hypotheses[0].statement == (
        "Random forest achieves higher balanced accuracy than logistic regression."
    )
    assert not any("real-time" in q for q in proto.research_questions)
    assert proto.research_questions[0].startswith(
        "Does random forest achieve higher balanced accuracy"
    )
    assert any("removed" in n for n in notes)


def test_metric_only_goal_updates_the_default_hypotheses() -> None:
    base = default_protocol("p", "cicids2017-mlcsv-v1")
    goal = "Compare the four models. Primary metric: MCC."
    proto, _ = _enforce_intent(base, research_intent(goal, []), goal)
    assert {h.primary_metric for h in proto.hypotheses} == {"mcc"}


def test_off_topic_papers_score_zero() -> None:
    assert (
        relevance(
            Paper(title="Prediction of low birth weight using Random Forest", doi="x", year=2017)
        )
        == 0
    )
    assert (
        relevance(Paper(title="Stroke disease classification with XGBoost", doi="x", year=2025))
        == 0
    )
    assert (
        relevance(Paper(title="Network intrusion detection with deep learning", doi="x", year=2023))
        > 0
    )
    assert (
        relevance(
            Paper(
                title="A tabular benchmark", doi="x", year=2023, abstract="We detect DDoS attacks."
            )
        )
        > 0
    )
