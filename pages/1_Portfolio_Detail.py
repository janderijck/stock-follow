import streamlit as st
import sqlite3
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from tax_calculator import TaxCalculator

DB_PATH = Path("data.db")

st.set_page_config(page_title="Aandeel Details", page_icon="📈", layout="wide")


# --------- Database helpers ---------
def get_connection():
    """Maakt verbinding met de database."""
    conn = sqlite3.connect(DB_PATH)

    # Zorg dat transactions tabel bestaat
    conn.executescript("""
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
        taxes REAL DEFAULT 0,
        exchange_rate REAL DEFAULT 1.0,
        notes TEXT
    );
    """)

    # Voeg taxes kolom toe als die nog niet bestaat
    try:
        conn.execute("ALTER TABLE transactions ADD COLUMN taxes REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Kolom bestaat al

    # Voeg fees_currency kolom toe als die nog niet bestaat
    try:
        conn.execute("ALTER TABLE transactions ADD COLUMN fees_currency TEXT DEFAULT 'EUR'")
    except sqlite3.OperationalError:
        pass  # Kolom bestaat al

    conn.executescript("""

    CREATE TABLE IF NOT EXISTS dividends (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        isin TEXT NOT NULL,
        ex_date TEXT NOT NULL,
        bruto_amount REAL NOT NULL,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS price_cache (
        ticker TEXT PRIMARY KEY,
        current_price REAL NOT NULL,
        change_percent REAL,
        currency TEXT,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS broker_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        broker_name TEXT NOT NULL UNIQUE,
        w8ben_signed INTEGER DEFAULT 0,
        w8ben_expiry_date TEXT
    );
    """)

    return conn


def format_currency(amount, currency='EUR'):
    """Format een bedrag met het juiste currency symbool."""
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CHF': 'CHF ',
        'CAD': 'C$',
        'AUD': 'A$',
    }

    symbol = currency_symbols.get(currency, currency + ' ')

    if currency == 'EUR':
        return f"€{amount:.2f}"
    else:
        return f"{symbol}{amount:.2f}"


def get_stock_details(ticker):
    """Haalt alle details op voor een specifiek aandeel."""
    conn = get_connection()

    query = """
    SELECT
        ticker,
        isin,
        name,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as total_quantity,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price_per_share ELSE 0 END) as total_invested,
        SUM(CASE WHEN transaction_type = 'BUY' THEN fees ELSE 0 END) as total_fees,
        currency,
        (SELECT broker FROM transactions t2 WHERE t2.ticker = transactions.ticker ORDER BY date DESC LIMIT 1) as broker
    FROM transactions
    WHERE ticker = ?
    GROUP BY ticker, isin, name, currency
    """

    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()

    if not df.empty:
        df['avg_purchase_price'] = (df['total_invested'] + df['total_fees']) / df['total_quantity']
        df['total_invested_with_fees'] = df['total_invested'] + df['total_fees']
        return df.iloc[0]
    return None


def get_all_purchases(ticker):
    """Haalt alle aankopen op voor een ticker."""
    conn = get_connection()

    query = """
    SELECT
        id,
        strftime('%d/%m/%Y', date) as Datum,
        date as date,
        broker as Broker,
        transaction_type as Type,
        quantity as Aantal,
        price_per_share as 'Prijs/stuk',
        currency as Valuta,
        fees as Kosten,
        COALESCE(taxes, 0) as Taxen,
        COALESCE(exchange_rate, 1.0) as Wisselkoers,
        COALESCE(fees_currency, 'EUR') as fees_currency,
        (quantity * price_per_share) as TotaalAankoop,
        (quantity * price_per_share + fees + COALESCE(taxes, 0)) as Totaal,
        notes as Notities
    FROM transactions
    WHERE ticker = ?
    ORDER BY date DESC
    """

    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()

    return df


def get_broker_for_ticker(ticker):
    """Haal de meest recente broker op voor een ticker."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT broker
        FROM transactions
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 1
    """, (ticker,))

    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None


def add_dividend(ticker, isin, ex_date, bruto_amount, notes="", currency="EUR", tax_paid=True, received=False):
    """Voegt een handmatig ingevoerd dividend toe."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO dividends (ticker, isin, ex_date, bruto_amount, notes, currency, tax_paid, received)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, isin, ex_date.isoformat(), float(bruto_amount), notes, currency, 1 if tax_paid else 0, 1 if received else 0)
    )
    conn.commit()
    conn.close()


def get_ownership_period(ticker):
    """Bepaalt de periode waarin je het aandeel in bezit had/hebt."""
    conn = get_connection()

    query = """
    SELECT
        date,
        transaction_type,
        quantity,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END)
            OVER (ORDER BY date) as running_total
    FROM transactions
    WHERE ticker = ?
    ORDER BY date
    """

    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()

    if df.empty:
        return None, None

    # Eerste aankoop
    first_purchase = df.iloc[0]['date']

    # Check of er een verkoop is waarbij alle aandelen zijn verkocht
    last_sale_date = None
    for idx, row in df.iterrows():
        if row['running_total'] == 0:
            last_sale_date = row['date']

    return first_purchase, last_sale_date


def get_manual_dividends(ticker, filter_ownership=False):
    """
    Haalt alle handmatig ingevoerde dividenden op voor een ticker.
    Als filter_ownership=True, filter alleen dividenden in de ownership periode.
    """
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT id, ex_date, bruto_amount, notes, currency, tax_paid, received,
               withheld_amount, additional_tax_due, net_received
        FROM dividends
        WHERE ticker = ?
        ORDER BY ex_date DESC
        """,
        conn,
        params=(ticker,)
    )
    conn.close()

    if filter_ownership and not df.empty:
        first_purchase, last_sale = get_ownership_period(ticker)

        if first_purchase:
            # Filter dividenden die binnen de ownership periode vallen
            df['ex_date'] = pd.to_datetime(df['ex_date'])
            first_purchase_dt = pd.to_datetime(first_purchase)

            # Filter: ex_date >= eerste aankoop
            df = df[df['ex_date'] >= first_purchase_dt]

            # Als er een volledige verkoop is geweest, filter ook tot die datum
            if last_sale:
                last_sale_dt = pd.to_datetime(last_sale)
                df = df[df['ex_date'] <= last_sale_dt]

            # Convert ex_date terug naar string voor display (dd/mm/yyyy)
            df['ex_date'] = df['ex_date'].dt.strftime('%d/%m/%Y')

    return df


def delete_dividend(dividend_id):
    """Verwijdert een dividend entry."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM dividends WHERE id = ?", (dividend_id,))
    conn.commit()
    conn.close()


def update_dividend(dividend_id, ex_date, bruto_amount, currency, tax_paid, received, notes=""):
    """Update een bestaand dividend."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE dividends
        SET ex_date = ?, bruto_amount = ?, currency = ?, tax_paid = ?, received = ?, notes = ?
        WHERE id = ?
        """,
        (ex_date.isoformat(), float(bruto_amount), currency, 1 if tax_paid else 0, 1 if received else 0, notes, dividend_id)
    )
    conn.commit()
    conn.close()


def calculate_total_dividends(ticker, filter_ownership=False):
    """Berekent totaal ontvangen dividend (bruto en netto) voor een ticker - ALLEEN ontvangen dividenden."""
    df = get_manual_dividends(ticker, filter_ownership=filter_ownership)

    if df.empty:
        return {
            'total_bruto': 0,
            'total_tax': 0,
            'total_netto': 0,
            'count': 0,
            'received_count': 0,
            'currency': 'EUR'
        }

    # Filter alleen ontvangen dividenden
    received_df = df[df['received'] == 1]

    if received_df.empty:
        return {
            'total_bruto': 0,
            'total_tax': 0,
            'total_netto': 0,
            'count': len(df),
            'received_count': 0,
            'currency': df['currency'].iloc[0] if 'currency' in df.columns and len(df) > 0 else 'EUR'
        }

    # Use intelligent tax calculator
    tax_calc = TaxCalculator()
    total_bruto = received_df['bruto_amount'].sum()
    total_tax = 0
    total_netto = 0

    for _, row in received_df.iterrows():
        tax_paid = row.get('tax_paid', 1)
        received = row.get('received', 0)

        # BELANGRIJKE REGEL: Tax kan alleen betaald zijn als dividend ontvangen is
        if not received:
            tax_paid = 0

        if tax_paid:
            # Check voor handmatige tax waarden
            withheld = row.get('withheld_amount', 0) or 0
            additional_tax = row.get('additional_tax_due', 0) or 0
            net_received_manual = row.get('net_received')

            if withheld > 0 or additional_tax > 0 or net_received_manual:
                # Gebruik handmatige waarden
                total_tax += (withheld + additional_tax)
                total_netto += (net_received_manual if net_received_manual else (row['bruto_amount'] - withheld - additional_tax))
            else:
                # Gebruik automatische calculator
                broker = get_broker_for_ticker(ticker)
                tax_result = tax_calc.calculate_tax(row['bruto_amount'], ticker, broker)
                total_tax += tax_result['total_tax']
                total_netto += tax_result['net_amount']
        else:
            # Geen tax betaald
            total_netto += row['bruto_amount']

    # Bepaal meest voorkomende currency
    div_currency = received_df['currency'].mode().iloc[0] if 'currency' in received_df.columns and len(received_df) > 0 else 'EUR'

    return {
        'total_bruto': total_bruto,
        'total_tax': total_tax,
        'total_netto': total_netto,
        'count': len(df),
        'received_count': len(received_df),
        'currency': div_currency
    }


def get_current_price(ticker, isin=None):
    """Haalt huidige prijs op via cache of yfinance."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT current_price, change_percent, currency FROM price_cache WHERE ticker = ?",
        (ticker,)
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            'current_price': result[0],
            'change_percent': result[1],
            'currency': result[2]
        }
    return None


def insert_transaction(date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, taxes=0, exchange_rate=1.0, notes=""):
    """Voegt een transactie toe aan de database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO transactions (date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, taxes, exchange_rate, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (date.isoformat(), broker, transaction_type, name, ticker.upper(), isin.upper(), int(quantity), float(price_per_share), currency, float(fees), float(taxes), float(exchange_rate), notes),
    )
    conn.commit()
    conn.close()


def update_transaction(transaction_id, date, broker, transaction_type, quantity, price_per_share, currency, fees, taxes=0, exchange_rate=1.0, notes="", fees_currency="EUR"):
    """Update een bestaande transactie."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE transactions
        SET date = ?, broker = ?, transaction_type = ?, quantity = ?,
            price_per_share = ?, currency = ?, fees = ?, taxes = ?, exchange_rate = ?, notes = ?, fees_currency = ?
        WHERE id = ?
        """,
        (date.isoformat(), broker, transaction_type, int(quantity), float(price_per_share),
         currency, float(fees), float(taxes), float(exchange_rate), notes, fees_currency, transaction_id),
    )
    conn.commit()
    conn.close()


def delete_transaction(transaction_id):
    """Verwijdert een transactie."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
    conn.commit()
    conn.close()


def get_transaction_by_id(transaction_id):
    """Haalt een specifieke transactie op."""
    conn = get_connection()
    query = """
    SELECT id, date, broker, transaction_type, name, ticker, isin,
           quantity, price_per_share, currency, fees, COALESCE(taxes, 0) as taxes,
           exchange_rate, notes, COALESCE(fees_currency, 'EUR') as fees_currency
    FROM transactions
    WHERE id = ?
    """
    df = pd.read_sql_query(query, conn, params=(transaction_id,))
    conn.close()

    if not df.empty:
        return df.iloc[0]
    return None


def get_available_brokers():
    """Haalt alle geconfigureerde brokers op."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT broker_name FROM broker_settings ORDER BY broker_name")
    brokers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return brokers


# --------- UI ---------
st.title("📈 Aandeel Details")

# Check if ticker is provided
if 'selected_ticker' not in st.session_state or not st.session_state.selected_ticker:
    st.warning("Geen aandeel geselecteerd. Ga terug naar het portfolio overzicht.")
    if st.button("← Terug naar Portfolio"):
        st.switch_page("pages/1_Portfolio.py")
    st.stop()

ticker = st.session_state.selected_ticker

# Terug knop
if st.button("← Terug naar Portfolio"):
    st.session_state.selected_ticker = None
    if 'view_detail_ticker' in st.query_params:
        del st.query_params['view_detail_ticker']
    st.switch_page("pages/1_Portfolio.py")

st.divider()

# Haal stock details op
stock_info = get_stock_details(ticker)

if stock_info is None:
    st.error(f"Geen gegevens gevonden voor ticker {ticker}")
    st.stop()

# Header met naam en ticker
st.header(f"{stock_info['name']} ({ticker})")

# Haal huidige prijs op
price_info = get_current_price(ticker, stock_info['isin'])
current_price = price_info['current_price'] if price_info else None

# Bereken performance (gebruik gefilterde dividenden)
div_info = calculate_total_dividends(ticker, filter_ownership=True)
if current_price:
    total_invested = stock_info['total_invested_with_fees']
    current_value = current_price * stock_info['total_quantity']
    gain_loss_excl = current_value - total_invested
    gain_loss_incl = gain_loss_excl + div_info['total_netto']
    gain_loss_percent = (gain_loss_incl / total_invested * 100) if total_invested > 0 else 0

# Toon key metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Broker", stock_info['broker'])

with col2:
    st.metric("Aantal", int(stock_info['total_quantity']))

with col3:
    if current_price:
        st.metric("Huidige Prijs", f"€{current_price:.2f}")
    else:
        st.metric("Huidige Prijs", "N/A")

with col4:
    st.metric("Avg Aankoopprijs", f"€{stock_info['avg_purchase_price']:.2f}")

with col5:
    st.metric("Totaal Geïnvesteerd", f"€{stock_info['total_invested_with_fees']:.2f}")

st.divider()

# Tweede rij metrics
if current_price:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Huidige Waarde", f"€{current_value:.2f}")

    with col2:
        st.metric("Dividenden (netto)", f"€{div_info['total_netto']:.2f}")

    with col3:
        st.metric(
            "W/V (excl. div)",
            f"€{gain_loss_excl:.2f}",
            delta=f"{(gain_loss_excl/total_invested*100):+.2f}%" if total_invested > 0 else None
        )

    with col4:
        st.metric(
            "W/V (incl. div)",
            f"€{gain_loss_incl:.2f}",
            delta=f"{gain_loss_percent:+.2f}%"
        )

st.divider()

# Tabs voor verschillende secties
tab1, tab2, tab3 = st.tabs(["📜 Transacties", "💰 Dividenden", "➕ Nieuwe Transactie"])

with tab1:
    st.subheader("Transactie Geschiedenis")

    purchases_df = get_all_purchases(ticker)

    if not purchases_df.empty:
        # Statistieken bovenaan
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            buy_count = len(purchases_df[purchases_df['Type'] == 'BUY'])
            st.metric("Aantal aankopen", buy_count)

        with col2:
            sell_count = len(purchases_df[purchases_df['Type'] == 'SELL'])
            st.metric("Aantal verkopen", sell_count)

        with col3:
            total_fees = purchases_df['Kosten'].sum()
            st.metric("Totale kosten", f"€{total_fees:.2f}")

        with col4:
            total_taxes = purchases_df['Taxen'].sum()
            st.metric("Totale taxen", f"€{total_taxes:.2f}")

        st.divider()

        # Initialize edit state
        if 'editing_transaction' not in st.session_state:
            st.session_state.editing_transaction = None

        # Header rij
        header_col1, header_col2, header_col3, header_col4, header_col5, header_col6, header_col7 = st.columns([1.5, 1, 1, 1.2, 1.2, 1.5, 0.5])
        with header_col1:
            st.markdown("**Datum**")
        with header_col2:
            st.markdown("**Type**")
        with header_col3:
            st.markdown("**Aantal**")
        with header_col4:
            st.markdown("**Prijs/stuk**")
        with header_col5:
            st.markdown("**Kosten + Tax**")
        with header_col6:
            st.markdown("**Totaal (Broker)**")
        with header_col7:
            st.markdown("")

        st.divider()

        # Toon elke transactie als een rij met edit knop
        for idx, row in purchases_df.iterrows():
            tx_id = row['id']
            is_editing = st.session_state.editing_transaction == tx_id

            if not is_editing:
                # View mode - toon transactie met edit knop
                currency = row.get('Valuta', 'EUR')
                currency_symbol = '$' if currency == 'USD' else '€' if currency == 'EUR' else '£' if currency == 'GBP' else currency
                taxes = row.get('Taxen', 0)
                fees = row.get('Kosten', 0)
                fees_currency = row.get('fees_currency', 'EUR') or 'EUR'
                fees_symbol = '$' if fees_currency == 'USD' else '€' if fees_currency == 'EUR' else '£'

                col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5, 1, 1, 1.2, 1.2, 1.5, 0.5])

                with col1:
                    st.write(f"**{row['Datum']}**")
                with col2:
                    type_emoji = "🟢" if row['Type'] == 'BUY' else "🔴"
                    st.write(f"{type_emoji} {row['Type']}")
                with col3:
                    st.write(f"{row['Aantal']} st")
                with col4:
                    # Prijs in originele valuta
                    st.write(f"{currency_symbol}{row['Prijs/stuk']:.2f}")
                with col5:
                    # Kosten en taxen in hun valuta
                    st.write(f"{fees_symbol}{fees:.2f} + {fees_symbol}{taxes:.2f}")
                with col6:
                    # Totaal aankoop in originele valuta
                    st.write(f"**{currency_symbol}{row['TotaalAankoop']:.2f}** ({row['Broker']})")
                with col7:
                    if st.button("✏️", key=f"edit_{tx_id}", help="Bewerk transactie"):
                        st.session_state.editing_transaction = tx_id
                        st.rerun()

                st.divider()

            else:
                # Edit mode - toon formulier
                tx_data = get_transaction_by_id(tx_id)

                with st.container():
                    st.info("✏️ Bezig met bewerken...")

                    with st.form(f"edit_form_{tx_id}"):
                        edit_col1, edit_col2, edit_col3 = st.columns(3)

                        with edit_col1:
                            edit_date = st.date_input("Datum", value=pd.to_datetime(tx_data['date']), key=f"date_{tx_id}", format="DD/MM/YYYY")
                            edit_type = st.selectbox("Type", ["BUY", "SELL"],
                                                    index=0 if tx_data['transaction_type'] == 'BUY' else 1,
                                                    key=f"type_{tx_id}")

                        with edit_col2:
                            edit_quantity = st.number_input("Aantal", min_value=1, step=1,
                                                           value=int(tx_data['quantity']),
                                                           key=f"qty_{tx_id}")
                            edit_price = st.number_input("Prijs/stuk", min_value=0.0, step=0.01, format="%.2f",
                                                        value=float(tx_data['price_per_share']),
                                                        key=f"price_{tx_id}")

                            # Valuta selectie
                            currency_options = ["EUR", "USD", "GBP"]
                            current_currency = tx_data.get('currency', 'EUR')
                            if current_currency in currency_options:
                                currency_idx = currency_options.index(current_currency)
                            else:
                                currency_idx = 0
                            edit_currency = st.selectbox("Valuta", currency_options,
                                                        index=currency_idx,
                                                        key=f"currency_{tx_id}")

                        with edit_col3:
                            # Valuta voor kosten/taxen
                            fees_currency_options = ["EUR", "USD", "GBP"]
                            current_fees_currency = tx_data.get('fees_currency', 'EUR') or 'EUR'
                            if current_fees_currency in fees_currency_options:
                                fees_currency_idx = fees_currency_options.index(current_fees_currency)
                            else:
                                fees_currency_idx = 0
                            edit_fees_currency = st.selectbox("Valuta kosten", fees_currency_options,
                                                              index=fees_currency_idx,
                                                              key=f"fees_currency_{tx_id}")

                            fees_symbol = '€' if edit_fees_currency == 'EUR' else '$' if edit_fees_currency == 'USD' else '£'
                            edit_fees = st.number_input(f"Kosten ({fees_symbol})", min_value=0.0, step=0.01, format="%.2f",
                                                       value=float(tx_data['fees']),
                                                       key=f"fees_{tx_id}",
                                                       help=f"Broker kosten in {edit_fees_currency}")
                            edit_taxes = st.number_input(f"Taxen ({fees_symbol})", min_value=0.0, step=0.01, format="%.2f",
                                                        value=float(tx_data.get('taxes', 0)),
                                                        key=f"taxes_{tx_id}",
                                                        help=f"Beurstaks in {edit_fees_currency}")

                            # Wisselkoers veld (altijd tonen, maar hint geven)
                            edit_exchange_rate = st.number_input(
                                "Wisselkoers (EUR/USD)",
                                min_value=0.0,
                                step=0.0001,
                                format="%.4f",
                                value=float(tx_data.get('exchange_rate', 1.0)),
                                key=f"exrate_{tx_id}",
                                help="Wisselkoers op moment van aankoop (bijv. 0.93 voor USD→EUR). Laat op 1.0 voor EUR."
                            )

                            # Broker dropdown - ALLEEN geconfigureerde brokers
                            available_brokers = get_available_brokers()
                            if available_brokers:
                                # Zoek huidige broker in de lijst
                                if tx_data['broker'] in available_brokers:
                                    current_broker_idx = available_brokers.index(tx_data['broker'])
                                else:
                                    # Broker bestaat niet meer in settings, gebruik eerste
                                    current_broker_idx = 0
                                    st.warning(f"⚠️ Broker '{tx_data['broker']}' niet gevonden in Broker Settings")

                                edit_broker = st.selectbox("Broker", available_brokers,
                                                          index=current_broker_idx,
                                                          key=f"broker_{tx_id}")
                            else:
                                st.error("⚠️ Geen brokers geconfigureerd. Ga naar **Broker Settings**")
                                edit_broker = tx_data['broker']  # Behoud huidige waarde

                        edit_notes = st.text_area("Notities", value=tx_data['notes'] if pd.notna(tx_data['notes']) else "",
                                                 height=60, key=f"notes_{tx_id}")

                        col_save, col_cancel, col_delete = st.columns([1, 1, 1])

                        with col_save:
                            if st.form_submit_button("💾 Opslaan", type="primary", use_container_width=True):
                                update_transaction(
                                    tx_id,
                                    edit_date,
                                    edit_broker,
                                    edit_type,
                                    edit_quantity,
                                    edit_price,
                                    edit_currency,
                                    edit_fees,
                                    edit_taxes,
                                    edit_exchange_rate,
                                    edit_notes,
                                    edit_fees_currency
                                )
                                st.session_state.editing_transaction = None
                                st.success("✓ Transactie bijgewerkt!")
                                st.rerun()

                        with col_cancel:
                            if st.form_submit_button("✗ Annuleer", use_container_width=True):
                                st.session_state.editing_transaction = None
                                st.rerun()

                        with col_delete:
                            if st.form_submit_button("🗑️ Verwijder", type="secondary", use_container_width=True):
                                delete_transaction(tx_id)
                                st.session_state.editing_transaction = None
                                st.success("Transactie verwijderd!")
                                st.rerun()

                st.divider()

    else:
        st.info("Nog geen transacties gevonden")

with tab2:
    st.subheader("💰 Dividend Beheer")

    # Toon ownership periode info
    first_purchase, last_sale = get_ownership_period(ticker)
    if first_purchase:
        if last_sale:
            st.info(f"📅 Aandeel in bezit van {first_purchase} tot {last_sale} (volledig verkocht)")
        else:
            st.info(f"📅 Aandeel in bezit sinds {first_purchase}")

    # Haal bestaande dividenden op (gefilterd op ownership periode)
    dividends_df = get_manual_dividends(ticker, filter_ownership=True)

    if not dividends_df.empty:
        # Bereken totalen met intelligente tax calculator
        tax_calc = TaxCalculator()
        broker = get_broker_for_ticker(ticker)

        # Initialize edit state
        if 'editing_dividend' not in st.session_state:
            st.session_state.editing_dividend = None

        # Header rij
        header_col1, header_col2, header_col3, header_col4, header_col5, header_col6, header_col7, header_col8 = st.columns([1.2, 1, 1, 1, 1, 1, 0.8, 0.5])
        with header_col1:
            st.markdown("**Datum**")
        with header_col2:
            st.markdown("**Bruto**")
        with header_col3:
            st.markdown("**🇺🇸 US Tax**")
        with header_col4:
            st.markdown("**🇧🇪 BE Tax**")
        with header_col5:
            st.markdown("**Netto**")
        with header_col6:
            st.markdown("**Ontv ✓**")
        with header_col7:
            st.markdown("")
        with header_col8:
            st.markdown("")

        st.divider()

        # Toon elke dividend als een rij met edit knop
        for _, row in dividends_df.iterrows():
            div_id = row['id']
            is_editing = st.session_state.editing_dividend == div_id

            bruto = row['bruto_amount']
            currency = row.get('currency', 'EUR')
            tax_paid = row.get('tax_paid', 1)
            received = row.get('received', 0)

            # Haal handmatige tax waarden op
            us_tax = row.get('withheld_amount', 0) or 0
            be_tax = row.get('additional_tax_due', 0) or 0
            net_received = row.get('net_received')

            # BELANGRIJKE REGEL: Tax kan alleen betaald zijn als dividend ontvangen is
            if not received:
                tax_paid = 0

            # Bereken netto
            if net_received:
                netto = net_received
            elif us_tax > 0 or be_tax > 0:
                netto = bruto - us_tax - be_tax
            elif tax_paid:
                tax_result = tax_calc.calculate_tax(bruto, ticker, broker)
                us_tax = tax_result.get('us_withholding', 0)
                be_tax = tax_result.get('belgian_tax', 0)
                netto = tax_result['net_amount']
            else:
                netto = bruto

            if not is_editing:
                # View mode
                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1.2, 1, 1, 1, 1, 1, 0.8, 0.5])

                with col1:
                    st.write(f"**{row['ex_date']}**")
                with col2:
                    st.write(format_currency(bruto, currency))
                with col3:
                    # US Tax in originele currency met betaald indicator
                    if us_tax > 0:
                        st.write(f"{format_currency(us_tax, currency)} ✅")
                    else:
                        st.write("-")
                with col4:
                    # BE Tax in EUR met betaald indicator
                    if be_tax > 0:
                        st.write(f"€{be_tax:.2f} ✅")
                    else:
                        st.write("-")
                with col5:
                    st.write(format_currency(netto, currency))
                with col6:
                    st.write("✅" if received else "❌")
                with col7:
                    st.write("")
                with col8:
                    if st.button("✏️", key=f"edit_div_{div_id}", help="Bewerk dividend"):
                        st.session_state.editing_dividend = div_id
                        st.rerun()

                st.divider()

            else:
                # Edit mode
                with st.container():
                    st.info("✏️ Bezig met bewerken...")

                    with st.form(f"edit_div_form_{div_id}"):
                        div_col1, div_col2 = st.columns(2)

                        with div_col1:
                            edit_div_date = st.date_input("Ex-Dividend Datum",
                                                          value=pd.to_datetime(row['ex_date']),
                                                          key=f"div_date_{div_id}",
                                                          format="DD/MM/YYYY")
                            edit_div_amount = st.number_input("Bruto bedrag", min_value=0.0, step=0.01,
                                                             format="%.2f", value=float(bruto),
                                                             key=f"div_amount_{div_id}")
                            edit_div_currency = st.selectbox("Currency",
                                                            options=["EUR", "USD", "GBP", "CHF", "CAD", "AUD", "JPY"],
                                                            index=["EUR", "USD", "GBP", "CHF", "CAD", "AUD", "JPY"].index(currency) if currency in ["EUR", "USD", "GBP", "CHF", "CAD", "AUD", "JPY"] else 0,
                                                            key=f"div_curr_{div_id}")

                        with div_col2:
                            edit_div_notes = st.text_area("Notities",
                                                         value=row['notes'] if pd.notna(row['notes']) else "",
                                                         height=40,
                                                         key=f"div_notes_{div_id}")
                            edit_div_received = st.checkbox("✅ Ontvangen",
                                                           value=bool(received),
                                                           key=f"div_recv_{div_id}")
                            edit_div_tax_paid = st.checkbox("💰 Tax betaald",
                                                           value=bool(tax_paid),
                                                           key=f"div_tax_{div_id}")

                        st.info("ℹ️ Tax kan alleen betaald zijn als dividend ontvangen is")

                        col_save, col_cancel, col_delete = st.columns([1, 1, 1])

                        with col_save:
                            if st.form_submit_button("💾 Opslaan", type="primary", use_container_width=True):
                                final_tax_paid = edit_div_tax_paid and edit_div_received
                                update_dividend(
                                    div_id,
                                    edit_div_date,
                                    edit_div_amount,
                                    edit_div_currency,
                                    final_tax_paid,
                                    edit_div_received,
                                    edit_div_notes
                                )
                                st.session_state.editing_dividend = None
                                st.success("✓ Dividend bijgewerkt!")
                                st.rerun()

                        with col_cancel:
                            if st.form_submit_button("✗ Annuleer", use_container_width=True):
                                st.session_state.editing_dividend = None
                                st.rerun()

                        with col_delete:
                            if st.form_submit_button("🗑️ Verwijder", type="secondary", use_container_width=True):
                                delete_dividend(div_id)
                                st.session_state.editing_dividend = None
                                st.success("Dividend verwijderd!")
                                st.rerun()

                st.divider()

        # Toon totalen
        st.divider()
        st.write(f"### 📊 Ontvangen Dividenden ({div_info['received_count']}/{div_info['count']})")
        col1, col2, col3 = st.columns(3)

        # Bepaal currency symbool
        div_currency = div_info.get('currency', 'EUR')
        div_symbol = '$' if div_currency == 'USD' else '€' if div_currency == 'EUR' else '£'

        with col1:
            st.metric("Totaal Bruto (ontvangen)", f"{div_symbol}{div_info['total_bruto']:.2f}")

        with col2:
            st.metric("Totaal Tax Betaald", f"{div_symbol}{div_info['total_tax']:.2f}")

        with col3:
            st.metric("Totaal Netto (ontvangen)", f"{div_symbol}{div_info['total_netto']:.2f}")
    else:
        st.info("Nog geen dividenden toegevoegd.")

    # Formulier om dividend toe te voegen
    st.divider()
    st.subheader("➕ Nieuw Dividend Toevoegen")

    # Haal laatste dividend per share op van yfinance
    default_div_per_share = 0.0
    default_bruto = 0.0
    default_us_tax = 0.0
    div_currency_default = 'USD' if stock_info.get('currency', 'EUR') == 'USD' else 'EUR'

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Probeer laatste dividend te halen
        last_dividend = info.get('lastDividendValue', 0) or info.get('dividendRate', 0) or 0
        if last_dividend and last_dividend > 0:
            # Voor kwartaal dividend, deel door 4 als het jaarlijks is
            div_frequency = info.get('dividendYield', 0)
            default_div_per_share = last_dividend

            # Bereken totaal bruto
            quantity = stock_info.get('total_quantity', 0)
            default_bruto = round(default_div_per_share * quantity, 2)

            # Bereken 15% US bronheffing
            default_us_tax = round(default_bruto * 0.15, 2)

            # Haal currency op
            div_currency_default = info.get('currency', 'USD')
    except:
        pass

    with st.form("dividend_form"):
        # Toon dividend info als beschikbaar
        if default_div_per_share > 0:
            symbol = '$' if div_currency_default == 'USD' else '€' if div_currency_default == 'EUR' else '£' if div_currency_default == 'GBP' else ''
            st.info(f"💡 Laatste dividend: {symbol}{default_div_per_share:.2f}/aandeel × {int(stock_info.get('total_quantity', 0))} = {symbol}{default_bruto:.2f}")

        col1, col2 = st.columns(2)

        with col1:
            div_date = st.date_input("Ex-Dividend Datum", format="DD/MM/YYYY")
            div_amount = st.number_input(
                "Bruto bedrag",
                min_value=0.0,
                value=default_bruto,
                step=0.01,
                format="%.2f",
                help="Voer het totale bruto dividend in (voor al je aandelen)"
            )
            currency_options = ["EUR", "USD", "GBP", "CHF", "CAD", "AUD", "JPY"]
            currency_index = currency_options.index(div_currency_default) if div_currency_default in currency_options else 1
            div_currency = st.selectbox(
                "Currency",
                options=currency_options,
                index=currency_index
            )

        with col2:
            div_notes = st.text_area("Notities (optioneel)", height=40)
            div_received = st.checkbox(
                "✅ Ontvangen",
                value=True,
                help="Vink aan als je dit dividend al ontvangen hebt"
            )

        # Belasting sectie
        st.write("**💰 Belasting**")
        tax_col1, tax_col2 = st.columns(2)

        with tax_col1:
            us_withholding = st.number_input(
                "🇺🇸 US Bronheffing",
                min_value=0.0,
                value=default_us_tax,
                step=0.01,
                format="%.2f",
                help="15% voor W-8BEN, 30% zonder"
            )
            us_tax_paid = st.checkbox("US tax betaald", value=False)

        with tax_col2:
            be_tax = st.number_input(
                "🇧🇪 Belgische RV",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                help="30% roerende voorheffing"
            )
            be_tax_paid = st.checkbox("BE tax betaald", value=False)

        div_submitted = st.form_submit_button("💾 Dividend opslaan", type="primary")

        if div_submitted and div_amount > 0:
            # Bereken netto ontvangen
            total_tax = us_withholding + be_tax
            net_received = div_amount - total_tax

            # Tax is betaald als beide zijn aangevinkt (of als er geen tax is)
            final_tax_paid = (us_tax_paid or us_withholding == 0) and (be_tax_paid or be_tax == 0)

            add_dividend(
                ticker,
                stock_info['isin'],
                div_date,
                div_amount,
                div_notes,
                div_currency,
                final_tax_paid,
                div_received
            )

            # Update met de handmatige tax waarden
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE dividends
                SET withheld_amount = ?, additional_tax_due = ?, net_received = ?
                WHERE id = (
                    SELECT id FROM dividends
                    WHERE ticker = ? AND ex_date = ?
                    ORDER BY id DESC LIMIT 1
                )
            """, (us_withholding, be_tax, net_received, ticker, div_date.isoformat()))
            conn.commit()
            conn.close()

            st.success(f"✓ Dividend van {format_currency(div_amount, div_currency)} toegevoegd! Netto: {format_currency(net_received, div_currency)}")
            st.rerun()

with tab3:
    st.subheader("➕ Nieuwe Transactie Toevoegen")

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            tx_date = st.date_input("Datum", format="DD/MM/YYYY")

            # Broker dropdown - ALLEEN geconfigureerde brokers
            available_brokers = get_available_brokers()
            if available_brokers:
                tx_broker = st.selectbox("Broker", available_brokers)
            else:
                st.error("⚠️ Geen brokers geconfigureerd. Ga naar **Broker Settings**")
                tx_broker = None

            tx_type = st.selectbox("Type", ["BUY", "SELL"])
            tx_quantity = st.number_input("Aantal aandelen", min_value=1, step=1)

        with col2:
            tx_price = st.number_input(
                "Prijs per aandeel",
                min_value=0.0,
                step=0.01,
                format="%.2f"
            )
            tx_currency = st.selectbox(
                "Valuta",
                options=["EUR", "USD", "GBP"],
                index=0
            )
            tx_fees = st.number_input(
                "Kosten (€)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                help="Broker kosten in EUR"
            )
            tx_taxes = st.number_input(
                "Taxen (€)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                help="Beurstaks in EUR"
            )

            # Wisselkoers veld (alleen tonen als niet EUR)
            if tx_currency != 'EUR':
                tx_exchange_rate = st.number_input(
                    "Wisselkoers (EUR/USD)",
                    min_value=0.0,
                    step=0.0001,
                    format="%.4f",
                    value=1.0,
                    help="Wisselkoers op moment van aankoop (bijv. 0.93 voor USD→EUR)"
                )
            else:
                tx_exchange_rate = 1.0

            tx_notes = st.text_area("Notities (optioneel)", height=100)

        tx_submitted = st.form_submit_button("💾 Transactie opslaan", type="primary")

        if tx_submitted:
            if not (tx_broker and tx_quantity > 0 and tx_price > 0):
                st.error("Vul alle verplichte velden in.")
            else:
                insert_transaction(
                    tx_date,
                    tx_broker,
                    tx_type,
                    stock_info['name'],
                    ticker,
                    stock_info['isin'],
                    tx_quantity,
                    tx_price,
                    tx_currency,
                    tx_fees,
                    tx_taxes,
                    tx_exchange_rate,
                    tx_notes
                )
                st.success(f"✓ {tx_type} transactie opgeslagen!")
                st.rerun()
