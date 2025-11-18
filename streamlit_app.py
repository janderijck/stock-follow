import streamlit as st
import sqlite3
import pandas as pd
import requests
from pathlib import Path

DB_PATH = Path("data.db")
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"


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


def get_available_shares(ticker):
    """Berekent het aantal beschikbare aandelen voor een ticker."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as available
        FROM transactions
        WHERE ticker = ?
        """,
        (ticker.upper(),)
    )

    result = cursor.fetchone()
    conn.close()

    return result[0] if result[0] is not None else 0


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


# --------- ISIN Lookup ---------
def lookup_isin(isin_code):
    """
    Zoekt aandeel informatie op basis van ISIN via OpenFIGI API.
    Retourneert dict met name, ticker, exchange, currency.
    """
    if not isin_code or len(isin_code.strip()) == 0:
        return None

    try:
        headers = {'Content-Type': 'application/json'}
        jobs = [{'idType': 'ID_ISIN', 'idValue': isin_code.strip().upper()}]

        response = requests.post(
            url=OPENFIGI_URL,
            headers=headers,
            json=jobs,
            timeout=10
        )

        if response.status_code != 200:
            return {"error": f"API error: status {response.status_code}"}

        data = response.json()

        # Check if we got results
        if not data or len(data) == 0:
            return {"error": "Geen resultaten gevonden"}

        first_result = data[0]

        # Check for error in response
        if 'error' in first_result:
            return {"error": f"ISIN niet gevonden: {first_result.get('error', 'Unknown error')}"}

        # Extract data from first match
        if 'data' in first_result and len(first_result['data']) > 0:
            security = first_result['data'][0]

            return {
                'name': security.get('name', ''),
                'ticker': security.get('ticker', ''),
                'exchange': security.get('exchCode', ''),
                'currency': security.get('marketSector', '').split()[0] if security.get('marketSector') else 'EUR',
                'security_type': security.get('securityType', ''),
                'market_sector': security.get('marketSector', '')
            }
        else:
            return {"error": "Geen data beschikbaar voor dit ISIN"}

    except requests.exceptions.Timeout:
        return {"error": "API timeout - probeer het opnieuw"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Netwerk error: {str(e)}"}
    except Exception as e:
        return {"error": f"Onverwachte fout: {str(e)}"}


# --------- Streamlit UI ---------
st.set_page_config(page_title="Portfolio tracker", page_icon="📈")

# Migreer database bij startup
migrate_database()

st.title("Transactie invoeren")

# Initialize session state
if 'security_data' not in st.session_state:
    st.session_state.security_data = {}
if 'lookup_result' not in st.session_state:
    st.session_state.lookup_result = None

# ISIN Lookup Section
st.subheader("ISIN Opzoeken")
col_lookup1, col_lookup2 = st.columns([3, 1])

with col_lookup1:
    isin_lookup = st.text_input(
        "Voer ISIN in om automatisch gegevens op te halen",
        key="isin_lookup_input",
        placeholder="Bv. US0378331005 (Apple)"
    )

with col_lookup2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("🔍 Opzoeken", type="primary"):
        if isin_lookup:
            with st.spinner("Gegevens ophalen..."):
                result = lookup_isin(isin_lookup)

                if result and 'error' not in result:
                    st.session_state.lookup_result = {
                        'name': result.get('name', ''),
                        'ticker': result.get('ticker', ''),
                        'isin': isin_lookup.upper(),
                        'currency': result.get('currency', 'EUR')
                    }
                    st.success(f"✓ Gevonden: {result.get('name', 'Unknown')} ({result.get('ticker', 'N/A')})")
                elif result and 'error' in result:
                    st.error(result['error'])
                    st.session_state.lookup_result = None
                else:
                    st.error("Kon geen informatie ophalen voor dit ISIN")
                    st.session_state.lookup_result = None
        else:
            st.warning("Voer eerst een ISIN in")

# Show lookup result with button to use it
if st.session_state.lookup_result:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.info(f"📊 Gevonden: {st.session_state.lookup_result.get('name', 'N/A')} - {st.session_state.lookup_result.get('ticker', 'N/A')}")

    with col2:
        if st.button("✅ Gebruik deze gegevens", type="primary"):
            st.session_state.security_data = st.session_state.lookup_result.copy()
            st.session_state.lookup_result = None
            st.success("Gegevens overgenomen!")
            st.rerun()

# Show currently selected data
if st.session_state.security_data:
    st.success(f"✓ Geselecteerd: {st.session_state.security_data.get('name', 'N/A')} ({st.session_state.security_data.get('ticker', 'N/A')})")
    if st.button("🗑️ Wis selectie"):
        st.session_state.security_data = {}
        st.rerun()

st.divider()

# Transaction Form
with st.form("new_transaction"):
    col1, col2, col3 = st.columns(3)

    with col1:
        date = st.date_input("Datum")
        broker = st.text_input("Broker (bv. DEGIRO)")
        transaction_type = st.selectbox("Type", ["BUY", "SELL"])
        name = st.text_input(
            "Naam aandeel",
            value=st.session_state.security_data.get('name', ''),
            key="name_input"
        )

    with col2:
        ticker = st.text_input(
            "Ticker (bv. AAPL)",
            value=st.session_state.security_data.get('ticker', ''),
            key="ticker_input"
        )
        isin = st.text_input(
            "ISIN",
            value=st.session_state.security_data.get('isin', ''),
            key="isin_input"
        )
        quantity = st.number_input("Aantal aandelen", min_value=1, step=1)
        price_per_share = st.number_input(
            "Prijs per aandeel",
            min_value=0.0,
            step=0.01,
            format="%.2f",
        )

    with col3:
        # Get currency from session state or default to EUR
        default_currency = st.session_state.security_data.get('currency', 'EUR')
        currency_options = ["EUR", "USD", "GBP"]
        if default_currency not in currency_options:
            default_currency = "EUR"

        currency = st.selectbox(
            "Valuta",
            currency_options,
            index=currency_options.index(default_currency)
        )
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

    submitted = st.form_submit_button("💾 Opslaan")


if submitted:
    if not (broker and name and ticker and isin and quantity > 0):
        st.error("Vul alle verplichte velden in.")
    else:
        # Extra validatie voor SELL transacties
        if transaction_type == "SELL":
            available = get_available_shares(ticker)
            if available < quantity:
                st.error(f"❌ Onvoldoende aandelen! Je hebt {available} aandelen van {ticker}, maar probeert {quantity} te verkopen.")
            else:
                insert_transaction(date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, exchange_rate, notes)
                st.success(f"✓ SELL transactie opgeslagen. Resterende aandelen: {available - quantity}")
                # Clear session state after saving
                st.session_state.security_data = {}
        else:
            insert_transaction(date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, exchange_rate, notes)
            st.success("✓ Transactie opgeslagen.")
            # Clear session state after saving
            st.session_state.security_data = {}


st.subheader("Bewaarde transacties")

try:
    df = load_transactions()
    if df.empty:
        st.info("Nog geen transacties opgeslagen.")
    else:
        st.dataframe(df, use_container_width=True)
except Exception as e:
    st.warning(f"Kon nog geen transacties laden: {e}")