import pandas as pd


def get_safety_recommendations(
    root_cause,
    incident_type,
    risk_level,
    narrative=None,
):
    """Generate preliminary safety review prompts, not verified findings."""

    def normalize(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip().lower()

    cause = normalize(root_cause)
    incident = normalize(incident_type)
    risk = normalize(risk_level)
    narrative_text = normalize(narrative)
    text = " ".join([cause, incident, narrative_text])

    recommendations = []

    def add(message):
        if message not in recommendations:
            recommendations.append(message)

    if any(term in text for term in (
        "ergonomic", "strain", "musculoskeletal", "repetitive",
        "fixed position", "awkward posture", "elbow", "back pain",
    )):
        add(
            "Review workstation ergonomics, posture, force, repetition, "
            "and time spent in a fixed position."
        )
        add(
            "Assess whether workstation adjustments, task variation, "
            "or recovery breaks should be considered."
        )

    if any(term in text for term in (
        "design deficiency", "design weakness", "design defect",
        "equipment design", "no built-in", "guarding", "interlock",
    )):
        add(
            "Arrange a competent engineering review of the equipment design "
            "and the reported limitation in its controls or guarding."
        )
        add(
            "Review whether the equipment can be operated and stopped safely "
            "from the relevant working position."
        )

    if any(term in text for term in (
        "induction", "training", "not trained", "untrained",
        "competency", "inexperienced", "new employee",
    )):
        add(
            "Verify induction and task-specific training records, including "
            "whether required training was completed before the task."
        )
        add(
            "Review task competency assessment and the supervision available "
            "to the worker."
        )

    if any(term in text for term in (
        "ppe", "personal protective equipment", "protective gloves",
        "safety glasses", "respirator",
    )):
        add(
            "Review whether suitable PPE was available, correctly fitted, "
            "and used as required."
        )
        add("Check relevant PPE training, inspection, and replacement records.")

    if any(term in text for term in (
        "maintenance", "equipment failure", "mechanical failure",
        "malfunction", "defective", "breakdown",
    )):
        add(
            "Arrange a competent inspection of the relevant equipment "
            "and document the findings."
        )
        add(
            "Review maintenance history, defect reports, and applicable "
            "isolation arrangements."
        )

    if any(term in text for term in (
        "procedure", "human error", "work instruction",
        "standard operating procedure", "unsafe act",
    )):
        add(
            "Compare the applicable work procedure with actual task "
            "conditions and available investigation evidence."
        )
        add(
            "Review whether workload, supervision, or unclear instructions "
            "may have contributed."
        )

    if any(term in text for term in (
        "poor communication", "miscommunication",
        "communication failure", "handover",
    )):
        add(
            "Review task communication, handover arrangements, and whether "
            "hazards and responsibilities were clearly understood."
        )

    if incident or narrative_text:
        add(
            "Compare the narrative with available records, witness accounts, "
            "and investigation findings before confirming contributing factors."
        )

    if risk in {"high", "critical", "very high"}:
        add(
            "Prioritize qualified review of the recorded risk classification "
            "and supporting evidence."
        )

    if not recommendations:
        add(
            "Review incident evidence and assess whether additional "
            "preventive controls require consideration."
        )

    return recommendations
