
from io import BytesIO
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    Image as ReportImage,
)

from src.recommendations import get_safety_recommendations

MOCHA = colors.HexColor("#2B1D16")
TEAL = colors.HexColor("#0D9488")
AMBER = colors.HexColor("#C8792C")
CREAM = colors.HexColor("#F8F3E9")
GRID = colors.HexColor("#E9E0D4")


def _clean(value, limit=180):
    """Convert values to safe, readable report text."""
    if pd.isna(value):
        return "Not recorded"

    value = str(value).strip()

    if not value:
        return "Not recorded"

    return value[:limit]


def _find_column(df, candidates):
    """Find a matching column without depending on capitalization."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def _count_table(title, series, limit=10):
    """Build a summary table from a pandas Series."""
    rows = [[title, "Records"]]

    if series is not None and not series.empty:
        for label, count in series.head(limit).items():
            rows.append([_clean(label, 100), f"{int(count):,}"])

    if len(rows) == 1:
        rows.append(["No data available", "0"])

    return rows


def _make_chart(series, title, chart_type="bar"):
    """Create a chart image in memory for the PDF."""
    if series is None or series.empty:
        return None

    values = series.dropna().head(8)
    if values.empty:
        return None

    image_buffer = BytesIO()
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    try:
        if chart_type == "pie":
            ax.pie(
                values.values,
                labels=[str(value)[:18] for value in values.index],
                autopct="%1.0f%%",
                startangle=90,
                textprops={"fontsize": 8},
            )
        else:
            ax.barh(
                [str(value)[:24] for value in values.index][::-1],
                values.values[::-1],
            )
            ax.set_xlabel("Number of records")
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)

        ax.set_title(title, fontsize=11, pad=10)
        fig.tight_layout()
        fig.savefig(
            image_buffer,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
    finally:
        plt.close(fig)

    image_buffer.seek(0)
    return image_buffer


def build_safety_report(
    incident_df,
    project_name="P_129 Safety Intelligence",
):
    """
    Generate a PDF report in memory.

    Returns:
        bytes: PDF file contents ready for Streamlit download_button.
    """
    df = incident_df.copy()

    # Remove application-only helper fields.
    df = df.drop(
        columns=["_year", "_month"],
        errors="ignore",
    )

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="P_129 Industrial Safety Report",
        author="P_129 Safety Intelligence",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="P129Title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=MOCHA,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="P129Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=15,
            textColor=TEAL,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="P129Heading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=MOCHA,
            spaceBefore=12,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="P129Body",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=MOCHA,
        )
    )

    story = []

    story.append(Paragraph(project_name, styles["P129Title"]))
    story.append(
        Paragraph(
            "Industrial Safety Incident Analysis",
            styles["P129Subtitle"],
        )
    )
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["P129Body"],
        )
    )
    story.append(Spacer(1, 6 * mm))

    # Identify common fields across supported datasets.
    severity_col = _find_column(
        df, ["severity_level", "severity"]
    )
    risk_col = _find_column(
        df, ["risk_level", "predicted_risk_level"]
    )
    cause_col = _find_column(
        df, ["root_cause_category", "root_cause", "predicted_root_cause"]
    )
    department_col = _find_column(
        df, ["department"]
    )
    incident_type_col = _find_column(
        df, ["incident_type", "event_type"]
    )
    date_col = _find_column(
        df, ["incident_date", "date"]
    )
    lost_days_col = _find_column(
        df, ["lost_time_days", "lost_days"]
    )

    story.append(
        Paragraph("1. Executive Summary", styles["P129Heading"])
    )
    story.append(
        Paragraph(
            f"This report summarizes {len(df):,} incident records "
            "selected from the application's supplied dataset. "
            "Findings describe the selected records and should not "
            "be interpreted as verified industry-wide statistics.",
            styles["P129Body"],
        )
    )
    story.append(Spacer(1, 4 * mm))

    # KPI summary.
    kpis = [
        ["Indicator", "Value"],
        ["Incident records", f"{len(df):,}"],
    ]

    if severity_col:
        critical = df[severity_col].astype(str).str.strip().str.lower()
        critical_count = critical.isin(["critical"]).sum()

        kpis.append(
            ["Records labelled Critical", f"{critical_count:,}"]
        )

    if risk_col:
        risk = df[risk_col].astype(str).str.strip().str.lower()
        high_risk = risk.isin(["high", "critical"]).sum()

        kpis.append(
            ["Records labelled High/Critical risk", f"{high_risk:,}"]
        )

    if lost_days_col:
        lost_days = pd.to_numeric(
            df[lost_days_col], errors="coerce"
        )
        kpis.append(
            [
                "Total recorded lost-time days",
                f"{lost_days.sum():,.0f}",
            ]
        )

    kpi_table = Table(kpis, colWidths=[105 * mm, 55 * mm])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), MOCHA),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), CREAM),
                ("TEXTCOLOR", (0, 1), (-1, -1), MOCHA),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("PADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(kpi_table)

    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph("Visual Safety Insights", styles["P129Heading"])
    )
    chart_images = []

    if severity_col:
        severity_counts = (
            df[severity_col]
            .fillna("Not recorded")
            .astype(str)
            .replace("", "Not recorded")
            .value_counts()
        )
        chart_buffer = _make_chart(
            severity_counts,
            "Incident Severity",
        )
        if chart_buffer:
            chart_images.append(
                ReportImage(chart_buffer, width=80 * mm, height=43 * mm)
            )

    if risk_col:
        risk_counts = (
            df[risk_col]
            .fillna("Not recorded")
            .astype(str)
            .replace("", "Not recorded")
            .value_counts()
        )
        chart_buffer = _make_chart(
            risk_counts,
            "Risk Level",
            chart_type="pie",
        )
        if chart_buffer:
            chart_images.append(
                ReportImage(chart_buffer, width=80 * mm, height=43 * mm)
            )

    if chart_images:
        chart_table = Table(
            [chart_images],
            colWidths=[88 * mm] * len(chart_images),
        )
        chart_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(chart_table)

    # Category distributions.
    distributions = [
        ("2. Severity Distribution", severity_col),
        ("3. Risk Level Distribution", risk_col),
        ("4. Root Cause Categories", cause_col),
        ("5. Department Distribution", department_col),
        ("6. Incident Type Distribution", incident_type_col),
    ]

    for heading, column in distributions:
        story.append(Paragraph(heading, styles["P129Heading"]))

        if not column:
            story.append(
                Paragraph(
                    "This field is not available in the selected dataset.",
                    styles["P129Body"],
                )
            )
            continue

        counts = (
            df[column]
            .fillna("Not recorded")
            .astype(str)
            .replace("", "Not recorded")
            .value_counts()
        )

        table = Table(
            _count_table("Category", counts),
            colWidths=[110 * mm, 50 * mm],
            repeatRows=1,
        )

        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), CREAM),
                    ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                    ("PADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )

        story.append(table)

    if cause_col:
        cause_counts = (
            df[cause_col]
            .dropna()
            .astype(str)
            .value_counts()
        )
        top_cause = cause_counts.index[0] if not cause_counts.empty else None
    else:
        top_cause = None

    if incident_type_col:
        incident_counts = (
            df[incident_type_col]
            .dropna()
            .astype(str)
            .value_counts()
        )
        top_incident = (
            incident_counts.index[0]
            if not incident_counts.empty
            else None
        )
    else:
        top_incident = None

    if risk_col:
        risk_counts = (
            df[risk_col]
            .dropna()
            .astype(str)
            .value_counts()
        )
        top_risk = risk_counts.index[0] if not risk_counts.empty else None
    else:
        top_risk = None

    recommendations = get_safety_recommendations(
        root_cause=top_cause,
        incident_type=top_incident,
        risk_level=top_risk,
    )

    recommendation_section = [
        Paragraph(
            "7. Suggested Preventive Review Actions",
            styles["P129Heading"],
        ),
        Paragraph(
            "These suggestions are preliminary review prompts, "
            "not verified findings or approved procedures.",
            styles["P129Body"],
        ),
    ]
    for suggestion in recommendations:
        recommendation_section.append(
            Paragraph(
                "• " + _clean(suggestion, 500),
                styles["P129Body"],
            )
        )
        recommendation_section.append(Spacer(1, 2 * mm))
    recommendation_section.append(
        Paragraph(
            "A qualified safety professional must review "
            "all suggested actions before implementation.",
            styles["P129Body"],
        )
    )
    story.append(KeepTogether(recommendation_section))

    # Recent records table.
    story.append(PageBreak())
    story.append(
        Paragraph("8. Incident Record Sample", styles["P129Heading"])
    )
    story.append(
        Paragraph(
            "The following table contains up to 25 records from the "
            "currently selected dataset. It is a sample, not a full "
            "listing of all incidents.",
            styles["P129Body"],
        )
    )
    story.append(Spacer(1, 3 * mm))

    preferred = [
        date_col,
        department_col,
        incident_type_col,
        severity_col,
        risk_col,
        cause_col,
    ]
    columns = [col for col in preferred if col is not None]

    if not columns:
        columns = list(df.columns[:5])

    sample = df[columns].head(25).copy()

    headers = [str(col).replace("_", " ").title() for col in columns]
    table_data = [headers]

    for _, row in sample.iterrows():
        table_data.append(
            [_clean(value, 65) for value in row.tolist()]
        )

    # Landscape pages make multi-column incident tables easier to read.
    # Keep the table within the available portrait page width.
    available_width = 176 * mm
    col_width = available_width / max(len(columns), 1)

    sample_table = Table(
        table_data,
        colWidths=[col_width] * len(columns),
        repeatRows=1,
    )

    sample_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), MOCHA),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("BACKGROUND", (0, 1), (-1, -1), CREAM),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("WORDWRAP", (0, 0), (-1, -1), True),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(sample_table)
    story.append(Spacer(1, 6 * mm))

    story.append(
        Paragraph("Interpretation and Limitations", styles["P129Heading"])
    )
    story.append(
        Paragraph(
            "This document summarizes dataset fields and counts; it "
            "does not independently verify incident reports or establish "
            "causation. Any machine-learning predictions are estimates "
            "and must be reviewed by qualified safety personnel before "
            "operational decisions are made.",
            styles["P129Body"],
        )
    )

    def draw_footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(GRID)
        canvas.line(
            17 * mm,
            14 * mm,
            A4[0] - 17 * mm,
            14 * mm,
        )
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MOCHA)
        canvas.drawString(
            17 * mm,
            9 * mm,
            "P_129 Safety Intelligence | Internal review",
        )
        canvas.drawRightString(
            A4[0] - 17 * mm,
            9 * mm,
            f"Page {document.page}",
        )
        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=draw_footer,
        onLaterPages=draw_footer,
    )

    buffer.seek(0)
    return buffer.getvalue()
