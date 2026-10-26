
import streamlit as st

from src.database import initialize_database

st.title("P_129 — Database Setup")

try:
    initialize_database()
    st.success("Database tables created or already exist.")

except Exception:
    st.error("Database setup failed.")
    st.exception(Exception("Check the terminal for the detailed error."))
