"""Prompt templates for AI playbook generation."""

from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent(
    """
    You are a senior retention strategist for subscription and repeat-purchase businesses.
    Your job is to translate segment analytics into concrete, execution-ready actions.
    Be specific, practical, and concise. Avoid generic advice.
    Always return Markdown with clear headings and bullet points.
    """
).strip()


def build_segment_prompt(
    segment_name: str,
    user_count: int,
    revenue_share: float,
    avg_churn_risk: float,
    avg_recency: float,
    avg_frequency: float,
) -> str:
    """Build a segment-level prompt with required context and output structure."""
    return dedent(
        f"""
        Analyze this retention segment and produce an actionable playbook.

        ## Segment Context
        - Segment name: {segment_name}
        - Users in segment: {user_count}
        - Revenue share (%): {revenue_share:.2f}
        - Average churn risk (0-100): {avg_churn_risk:.2f}
        - Average recency (days since last order): {avg_recency:.2f}
        - Average frequency (orders per user): {avg_frequency:.2f}

        ## Instructions
        - Use only the context above and reason explicitly from the metrics.
        - Make recommendations specific to this segment, not generic lifecycle advice.
        - Prefer actions that can be executed in the next 30 days.
        - When useful, include campaign examples, channel suggestions, and simple prioritization.

        ## Output Format (Markdown, required)
        ### Diagnosis
        ### Hypotheses
        ### Tactical Actions
        ### Messaging Angles
        ### Metrics to Track
        ### Risks

        In each section, provide concrete and testable recommendations.
        """
    ).strip()


def build_summary_prompt(all_segments_context: str) -> str:
    """Build an optional cross-segment strategic summary prompt."""
    return dedent(
        f"""
        Build a cross-segment retention strategy summary based on the context below.

        ## Segments Context
        {all_segments_context}

        ## Instructions
        - Identify the highest-priority segments by combined revenue impact and churn risk.
        - Recommend a practical execution sequence (what to do first, second, third).
        - Call out trade-offs, dependencies, and risks.
        - Keep language business-friendly and decision-oriented.

        ## Output Format (Markdown, required)
        ### Strategic Priorities
        ### 30-Day Action Plan
        ### KPI Framework
        ### Risks and Mitigations
        """
    ).strip()
