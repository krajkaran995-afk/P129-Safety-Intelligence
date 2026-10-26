"""Neon PostgreSQL persistence helpers for P_129.

Requires:
    SQLAlchemy >= 2.0
    psycopg[binary] >= 3.2

Expected Streamlit secret in .streamlit/secrets.toml:
    [connections.neon]
    url = "postgresql://..."
"""
from __future__ import annotations

from typing import Any

import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

@st.cache_resource
def get_engine() -> Engine:
    """Create one pooled SQLAlchemy engine per Streamlit process."""
    database_url = st.secrets["connections"]["neon"]["url"]
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"connect_timeout": 10},
    )


def initialize_database() -> None:
    """Create the app tables if they do not exist yet."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id BIGSERIAL PRIMARY KEY,
            incident_date DATE NULL,
            department TEXT NULL,
            location TEXT NULL,
            incident_type TEXT NULL,
            severity_level TEXT NULL,
            risk_level TEXT NULL,
            description TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'user_submitted',
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id BIGSERIAL PRIMARY KEY,
            incident_id BIGINT NULL REFERENCES incidents(id) ON DELETE SET NULL,
            prediction_type TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            recorded_label TEXT NULL,
            model_name TEXT NOT NULL,
            incident_date DATE NULL,
            department TEXT NULL,
            location TEXT NULL,
            incident_type TEXT NULL,
            description TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id BIGSERIAL PRIMARY KEY,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id BIGINT NULL,
            details TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_predictions_created_at
        ON predictions (created_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_predictions_type
        ON predictions (prediction_type)
        """,
    ]
    engine = get_engine()
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def save_incident(
    *,
    description: str,
    incident_date: Any = None,
    department: str | None = None,
    location: str | None = None,
    incident_type: str | None = None,
    severity_level: str | None = None,
    risk_level: str | None = None,
    source: str = "user_submitted",
) -> int:
    """Insert a submitted incident and return its database ID."""
    if not description or not description.strip():
        raise ValueError("Incident description is required.")

    query = text(
        """
        INSERT INTO incidents
            (incident_date, department, location, incident_type,
             severity_level, risk_level, description, source)
        VALUES
            (:incident_date, :department, :location, :incident_type,
             :severity_level, :risk_level, :description, :source)
        RETURNING id
        """
    )
    values = {
        "incident_date": incident_date,
        "department": department,
        "location": location,
        "incident_type": incident_type,
        "severity_level": severity_level,
        "risk_level": risk_level,
        "description": description.strip(),
        "source": source,
    }
    with get_engine().begin() as connection:
        incident_id = connection.execute(query, values).scalar_one()
        connection.execute(
            text(
                """
                INSERT INTO audit_logs (action, entity_type, entity_id, details)
                VALUES ('create', 'incident', :entity_id, :details)
                """
            ),
            {"entity_id": incident_id, "details": f"source={source}"},
        )
    return int(incident_id)


def save_prediction(
    *,
    prediction_type: str,
    predicted_label: str,
    model_name: str,
    description: str,
    recorded_label: str | None = None,
    incident_id: int | None = None,
    incident_date: Any = None,
    department: str | None = None,
    location: str | None = None,
    incident_type: str | None = None,
) -> int:
    """Persist a model result and a corresponding audit event."""
    if prediction_type not in {"root_cause", "risk"}:
        raise ValueError("prediction_type must be 'root_cause' or 'risk'.")
    if not predicted_label.strip():
        raise ValueError("predicted_label is required.")
    if not description.strip():
        raise ValueError("description is required.")

    query = text(
        """
        INSERT INTO predictions
            (incident_id, prediction_type, predicted_label, recorded_label,
             model_name, incident_date, department, location, incident_type,
             description)
        VALUES
            (:incident_id, :prediction_type, :predicted_label, :recorded_label,
             :model_name, :incident_date, :department, :location, :incident_type,
             :description)
        RETURNING id
        """
    )
    values = {
        "incident_id": incident_id,
        "prediction_type": prediction_type,
        "predicted_label": predicted_label,
        "recorded_label": recorded_label,
        "model_name": model_name,
        "incident_date": incident_date,
        "department": department,
        "location": location,
        "incident_type": incident_type,
        "description": description.strip(),
    }
    with get_engine().begin() as connection:
        prediction_id = connection.execute(query, values).scalar_one()
        connection.execute(
            text(
                """
                INSERT INTO audit_logs (action, entity_type, entity_id, details)
                VALUES ('prediction_created', 'prediction', :entity_id, :details)
                """
            ),
            {
                "entity_id": prediction_id,
                "details": f"type={prediction_type}; model={model_name}",
            },
        )
    return int(prediction_id)


def get_prediction_history(limit: int = 100):
    """Return recent prediction history as a pandas DataFrame."""
    import pandas as pd

    safe_limit = max(1, min(int(limit), 500))
    query = text(
        """
        SELECT id, prediction_type, predicted_label, recorded_label, model_name,
               incident_date, department, location, incident_type, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    with get_engine().connect() as connection:
        return pd.read_sql_query(query, connection, params={"limit": safe_limit})
