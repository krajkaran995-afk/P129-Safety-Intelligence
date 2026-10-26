
import streamlit as st
from sqlalchemy import text

st.title("P_129 — Neon Database Test")

try:
    conn = st.connection("neon", type="sql")

    with conn.session as session:
        result = session.execute(
            text("SELECT current_database(), current_user")
        ).fetchone()

    st.success("Neon PostgreSQL connected successfully!")
    st.write("Database:", result[0])
    st.write("Role:", result[1])

except Exception as exc:
    st.error("Database connection failed.")
    st.exception(exc)
