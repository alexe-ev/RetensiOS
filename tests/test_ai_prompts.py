from __future__ import annotations

from rfm_engine.ai.prompts import SYSTEM_PROMPT, build_segment_prompt, build_summary_prompt


def test_system_prompt_sets_retention_strategist_persona() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "retention strategist" in prompt
    assert "markdown" in prompt


def test_build_segment_prompt_renders_required_context_values() -> None:
    rendered = build_segment_prompt(
        segment_name="Champions",
        user_count=250,
        revenue_share=38.25,
        avg_churn_risk=22.1,
        avg_recency=5.4,
        avg_frequency=7.8,
    )

    assert "Champions" in rendered
    assert "250" in rendered
    assert "38.25" in rendered
    assert "22.10" in rendered
    assert "5.40" in rendered
    assert "7.80" in rendered


def test_build_segment_prompt_requests_all_required_sections() -> None:
    rendered = build_segment_prompt(
        segment_name="At Risk",
        user_count=120,
        revenue_share=19.4,
        avg_churn_risk=75.0,
        avg_recency=34.0,
        avg_frequency=1.7,
    )

    assert "### Diagnosis" in rendered
    assert "### Hypotheses" in rendered
    assert "### Tactical Actions" in rendered
    assert "### Messaging Angles" in rendered
    assert "### Metrics to Track" in rendered
    assert "### Risks" in rendered


def test_build_segment_prompt_supports_all_standard_segment_names() -> None:
    segments = [
        "Champions",
        "Loyal",
        "Potential Loyalists",
        "New",
        "Promising",
        "At Risk",
        "Hibernating",
        "Lost",
    ]

    for segment_name in segments:
        rendered = build_segment_prompt(
            segment_name=segment_name,
            user_count=10,
            revenue_share=12.0,
            avg_churn_risk=50.0,
            avg_recency=14.0,
            avg_frequency=2.0,
        )
        assert segment_name in rendered


def test_build_segment_prompt_accepts_non_standard_segment_name() -> None:
    rendered = build_segment_prompt(
        segment_name="VIP Early Adopters",
        user_count=42,
        revenue_share=9.5,
        avg_churn_risk=31.2,
        avg_recency=8.0,
        avg_frequency=4.2,
    )
    assert "VIP Early Adopters" in rendered


def test_build_summary_prompt_includes_passed_context_and_structure() -> None:
    context = "- Champions: high revenue, low churn\n- At Risk: high revenue at risk"
    rendered = build_summary_prompt(context)

    assert context in rendered
    assert "### Strategic Priorities" in rendered
    assert "### 30-Day Action Plan" in rendered
    assert "### KPI Framework" in rendered
    assert "### Risks and Mitigations" in rendered
