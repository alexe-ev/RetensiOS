"""Executive report generation."""

from __future__ import annotations

import pandas as pd


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _format_date(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%Y-%m-%d")


def _build_segment_table(segment_summary: pd.DataFrame) -> str:
    header = (
        "| Segment | Users | Revenue Share (%) | Avg Churn Risk (%) | "
        "Avg Recency (days) | Revenue at Risk |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    rows: list[str] = []
    for row in segment_summary.itertuples(index=False):
        rows.append(
            "| "
            f"{row.segment} | "
            f"{int(row.user_count)} | "
            f"{float(row.revenue_share_pct):.2f} | "
            f"{float(row.avg_churn_risk):.2f} | "
            f"{float(row.avg_recency_days):.1f} | "
            f"{_format_currency(float(row.revenue_at_risk))} |"
        )
    return "\n".join([header, *rows])


def build_report(
    profile: dict,
    segment_summary: pd.DataFrame,
    concentration_index: float,
) -> str:
    """Build an executive retention report in deterministic Markdown."""
    if segment_summary.empty:
        raise ValueError("segment_summary must not be empty.")

    summary = (
        segment_summary.copy()
        .sort_values("revenue_share_pct", ascending=False)
        .reset_index(drop=True)
    )
    top_segment = summary.iloc[0]
    highest_risk_segment = summary.sort_values("avg_churn_risk", ascending=False).iloc[0]
    priority_segments = summary.sort_values("revenue_at_risk", ascending=False).head(3)
    total_users = int(summary["user_count"].sum())
    total_revenue = float(summary["total_revenue"].sum())
    total_revenue_at_risk = float(summary["revenue_at_risk"].sum())

    distribution_lines = []
    for row in summary.itertuples(index=False):
        distribution_lines.append(
            f"- {row.segment}: "
            f"{float(row.revenue_share_pct):.2f}% of revenue "
            f"({_format_currency(float(row.total_revenue))})"
        )

    concentration_share_text = (
        f"the top 2 segments account for {concentration_index:.2f}% of revenue."
    )
    concentration_note = (
        f"Revenue concentration is high: {concentration_share_text}"
        if concentration_index > 60
        else f"Revenue concentration is moderate: {concentration_share_text}"
    )

    top_risk_lines = []
    for row in priority_segments.itertuples(index=False):
        top_risk_lines.append(
            f"- {row.segment}: "
            f"{_format_currency(float(row.revenue_at_risk))} at risk "
            f"({float(row.avg_churn_risk):.2f}% average risk)"
        )
    strategic_priority = ", ".join(priority_segments["segment"].tolist())

    median_user_count = float(summary["user_count"].median())
    growth_candidates = summary[
        (summary["avg_churn_risk"] <= 30) & (summary["user_count"] >= median_user_count)
    ].sort_values(["user_count", "revenue_share_pct"], ascending=[False, False])
    recovery_candidates = summary[summary["revenue_at_risk"] > 0].sort_values(
        ["revenue_at_risk", "avg_churn_risk"],
        ascending=[True, False],
    )

    quick_wins: list[str] = []
    if not growth_candidates.empty:
        row = growth_candidates.iloc[0]
        quick_wins.append(
            f"- Growth potential: {row['segment']} has {int(row['user_count'])} users with "
            f"{float(row['avg_churn_risk']):.2f}% average risk."
        )
    if not recovery_candidates.empty:
        row = recovery_candidates.iloc[0]
        revenue_at_risk = _format_currency(float(row["revenue_at_risk"]))
        quick_wins.append(
            f"- Fast recovery candidate: {row['segment']} has {revenue_at_risk} "
            f"at risk and can be targeted with a focused retention campaign."
        )
    if not quick_wins:
        quick_wins.append("- No immediate quick-win segments detected from this snapshot.")

    date_min = _format_date(profile["date_min"])
    date_max = _format_date(profile["date_max"])

    summary_text = (
        f"This snapshot covers {total_users} users and "
        f"{_format_currency(total_revenue)} in total revenue. "
        f"The largest segment is {top_segment['segment']} "
        f"({float(top_segment['revenue_share_pct']):.2f}% share). "
        f"The highest average risk is in {highest_risk_segment['segment']} "
        f"({float(highest_risk_segment['avg_churn_risk']):.2f}%)."
    )
    strategic_priority_text = (
        "Prioritize the following segments first based on revenue exposure: "
        f"{strategic_priority}. "
        "These segments combine meaningful revenue contribution with elevated "
        "retention pressure."
    )

    sections = [
        "# RetensiOS Executive Retention Report",
        "## Executive Summary",
        summary_text,
        "## Revenue Distribution",
        "\n".join(distribution_lines),
        concentration_note,
        "## Risk Exposure",
        f"Total revenue at risk: {_format_currency(total_revenue_at_risk)}.",
        "\n".join(top_risk_lines),
        "## Segment Overview Table",
        _build_segment_table(summary),
        "## Strategic Priority",
        strategic_priority_text,
        "## Quick Wins",
        "\n".join(quick_wins),
        "---",
        f"Data window: {date_min} to {date_max}",
        f"Generated timestamp: {date_max} 00:00:00 UTC",
    ]
    return "\n\n".join(sections).strip() + "\n"
