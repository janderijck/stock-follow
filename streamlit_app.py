import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data.db")


# --------- Database helpers ---------
def migrate_database():
    """Migreert oude database schema naar nieuwe versie."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check of transactions tabel bestaat en welke kolommen het heeft
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
    table_exists = cursor.fetchone() is not None

    if table_exists:
        # Check of broker kolom bestaat (nieuwe schema)
        cursor.execute("PRAGMA table_info(transactions)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'broker' not in columns:
            # Oude schema detecteerd - migreer data
            print("Migrating database to new schema...")

            # Backup oude data
            cursor.execute("SELECT * FROM transactions")
            old_data = cursor.fetchall()

            # Drop oude tabel
            cursor.execute("DROP TABLE transactions")

            # Maak nieuwe tabel
            cursor.executescript("""
                CREATE TABLE transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    broker TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    isin TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price_per_share REAL NOT NULL,
                    currency TEXT NOT NULL,
                    fees REAL NOT NULL,
                    exchange_rate REAL DEFAULT 1.0,
                    notes TEXT
                );
            """)

            # Migreer oude data met default waarden voor nieuwe kolommen
            for row in old_data:
                # Oude schema: id, date, name, ticker, isin, price, fees
                old_id, old_date, old_name, old_ticker, old_isin, old_price, old_fees = row
                cursor.execute(
                    """INSERT INTO transactions
                    (date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, exchange_rate, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (old_date, "UNKNOWN", "BUY", old_name, old_ticker, old_isin, 1, old_price, "EUR", old_fees, 1.0, "Gemigreerd van oude data")
                )

            conn.commit()
            print(f"Migrated {len(old_data)} transactions to new schema")

    conn.close()


def get_connection():
    """Maakt (of opent) de SQLite database en zorgt dat de tabel bestaat."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    broker TEXT NOT NULL,
    transaction_type TEXT NOT NULL,
    name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    isin TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price_per_share REAL NOT NULL,
    currency TEXT NOT NULL,
    fees REAL NOT NULL,
    exchange_rate REAL DEFAULT 1.0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS cash_balance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    broker TEXT NOT NULL,
    currency TEXT NOT NULL,
    amount REAL NOT NULL,
    last_updated TEXT NOT NULL,
    UNIQUE(broker, currency)
);
        """
    )
    return conn


def insert_transaction(date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, exchange_rate=1.0, notes=""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO transactions (date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, exchange_rate, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (date.isoformat(), broker, transaction_type, name, ticker.upper(), isin.upper(), int(quantity), float(price_per_share), currency, float(fees), float(exchange_rate), notes),
    )
    conn.commit()
    conn.close()


def load_transactions():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT date AS Datum, broker AS Broker, transaction_type AS Type, "
        "name AS Aandeel, ticker AS Ticker, isin AS ISIN, "
        "quantity AS Aantal, price_per_share AS 'Prijs/stuk', currency AS Valuta, "
        "fees AS Kosten, exchange_rate AS Wisselkoers, notes AS Notities "
        "FROM transactions ORDER BY date DESC",
        conn,
    )
    conn.close()
    return df

# --------- Streamlit UI ---------
st.set_page_config(page_title="Portfolio tracker", page_icon="📈")

# Migreer database bij startup
migrate_database()

st.title("Transactie invoeren")

with st.form("new_transaction"):
    col1, col2, col3 = st.columns(3)

    with col1:
        date = st.date_input("Datum")
        broker = st.text_input("Broker (bv. DEGIRO)")
        transaction_type = st.selectbox("Type", ["BUY", "SELL"])
        name = st.text_input("Naam aandeel")

    with col2:
        ticker = st.text_input("Ticker (bv. AAPL)")
        isin = st.text_input("ISIN")
        quantity = st.number_input("Aantal aandelen", min_value=1, step=1)
        price_per_share = st.number_input(
            "Prijs per aandeel",
            min_value=0.0,
            step=0.01,
            format="%.2f",
        )

    with col3:
        currency = st.selectbox("Valuta", ["EUR", "USD", "GBP"])
        fees = st.number_input(
            "Kosten (broker + taksen)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
        )
        exchange_rate = st.number_input(
            "Wisselkoers naar EUR",
            min_value=0.0,
            value=1.0,
            step=0.01,
            format="%.4f",
        )
        notes = st.text_area("Notities (optioneel)", height=100)

    submitted = st.form_submit_button("Opslaan")


if submitted:
    if not (broker and name and ticker and isin and quantity > 0):
        st.error("Vul alle verplichte velden in.")
    else:
        insert_transaction(date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, exchange_rate, notes)
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