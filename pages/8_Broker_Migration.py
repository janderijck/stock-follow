import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data.db")

st.set_page_config(page_title="Broker Migratie", page_icon="🔄", layout="wide")

st.title("🔄 Broker Migratie Tool")

st.info("""
Deze tool helpt je om transacties met brokers die niet in Broker Settings staan te updaten.
Dit is handig als je oude transacties hebt met brokers die nog niet geconfigureerd zijn.
""")


def get_connection():
    """Maakt verbinding met de database."""
    return sqlite3.connect(DB_PATH)


def get_available_brokers():
    """Haalt alle geconfigureerde brokers op."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT broker_name FROM broker_settings ORDER BY broker_name")
    brokers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return brokers


def get_unconfigured_brokers():
    """Vindt alle brokers in transactions die niet in broker_settings staan."""
    conn = get_connection()

    query = """
    SELECT DISTINCT t.broker
    FROM transactions t
    WHERE t.broker NOT IN (SELECT broker_name FROM broker_settings)
    ORDER BY t.broker
    """

    cursor = conn.cursor()
    cursor.execute(query)
    brokers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return brokers


def count_transactions_by_broker(broker):
    """Telt het aantal transacties voor een broker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE broker = ?", (broker,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def migrate_broker(old_broker, new_broker):
    """Migreert alle transacties van een oude broker naar een nieuwe."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE transactions SET broker = ? WHERE broker = ?", (new_broker, old_broker))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected


# Haal data op
unconfigured = get_unconfigured_brokers()
configured = get_available_brokers()

if not configured:
    st.warning("⚠️ Geen brokers geconfigureerd in Broker Settings. Ga eerst naar **Broker Settings** om brokers toe te voegen.")
    st.stop()

if not unconfigured:
    st.success("✅ Alle brokers in je transacties zijn geconfigureerd in Broker Settings!")
    st.stop()

# Toon unconfigured brokers
st.subheader("Niet-geconfigureerde Brokers")
st.write(f"Gevonden **{len(unconfigured)}** broker(s) die niet in Broker Settings staan:")

for old_broker in unconfigured:
    count = count_transactions_by_broker(old_broker)

    with st.expander(f"**{old_broker}** ({count} transacties)", expanded=True):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.write(f"**Oude broker:** `{old_broker}`")
            st.write(f"**Aantal transacties:** {count}")

        with col2:
            new_broker = st.selectbox(
                "Migreer naar geconfigureerde broker:",
                configured,
                key=f"select_{old_broker}"
            )

        with col3:
            st.write("")
            st.write("")
            if st.button("🔄 Migreer", key=f"migrate_{old_broker}", type="primary"):
                affected = migrate_broker(old_broker, new_broker)
                st.success(f"✓ {affected} transacties gemigreerd van '{old_broker}' naar '{new_broker}'!")
                st.rerun()

st.divider()

# Optie om nieuwe broker toe te voegen
st.subheader("➕ Of voeg nieuwe broker toe")
st.write("Als je een nieuwe broker wilt toevoegen aan Broker Settings:")

if st.button("Ga naar Broker Settings", type="secondary"):
    st.switch_page("pages/6_Broker_Settings.py")
