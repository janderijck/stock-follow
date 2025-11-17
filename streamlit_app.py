import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data.db")


# --------- Database helpers ---------
def get_connection():
    """Maakt (of opent) de SQLite database en zorgt dat de tabel bestaat."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    broker TEXT NOT NULL,           -- NIEUW: welke broker
    transaction_type TEXT NOT NULL, -- NIEUW: 'BUY' of 'SELL'
    name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    isin TEXT NOT NULL,
    quantity INTEGER NOT NULL,      -- NIEUW: aantal aandelen
    price_per_share REAL NOT NULL,
    currency TEXT NOT NULL,         -- NIEUW: EUR, USD, etc.
    fees REAL NOT NULL,
    exchange_rate REAL DEFAULT 1.0, -- NIEUW: wisselkoers naar EUR
    notes TEXT
)
CREATE TABLE IF NOT EXISTS cash_balance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    broker TEXT NOT NULL,
    currency TEXT NOT NULL,
    amount REAL NOT NULL,
    last_updated TEXT NOT NULL,
    UNIQUE(broker, currency)
        )
        """
    )
    return conn


def insert_transaction(date, name, ticker, isin, price, fees):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO transactions (date, name, ticker, isin, price, fees)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (date.isoformat(), name, ticker.upper(), isin.upper(), float(price), float(fees)),
    )
    conn.commit()
    conn.close()


def load_transactions():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT date AS Datum, name AS Aandeel, ticker AS Ticker, "
        "isin AS ISIN, price AS Prijs, fees AS Kosten "
        "FROM transactions ORDER BY date",
        conn,
    )
    conn.close()
    return df

# --------- Streamlit UI ---------
st.set_page_config(page_title="Portfolio tracker", page_icon="📈")
st.title("Transactie invoeren")

with st.form("new_transaction"):
    col1, col2 = st.columns(2)

    with col1:
        date = st.date_input("Koopdatum")
        name = st.text_input("Naam aandeel")
        ticker = st.text_input("Ticker (bv. AAPL)")

    with col2:
        isin = st.text_input("ISIN")
        price = st.number_input(
            "Prijs per stuk",
            min_value=0.0,
            step=0.01,
            format="%.2f",
        )
        fees = st.number_input(
            "Kosten (broker + taksen)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
        )

    submitted = st.form_submit_button("Opslaan")


if submitted:
    if not (name and ticker and isin):
        st.error("Vul minstens naam, ticker en ISIN in.")
    else:
        insert_transaction(date, name, ticker, isin, price, fees)
        st.success("Transactie opgeslagen.")


st.subheader("Bewaarde transacties")

try:
    df = load_transactions()
    if df.empty:
        st.info("Nog geen transacties opgeslagen.")
    else:
        st.dataframe(df, use_container_width=True)
except Exception as e:
    st.warning(f"Kon nog geen transacties laden: {e}")