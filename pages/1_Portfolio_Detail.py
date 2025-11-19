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
        exchange_rate REAL DEFAULT 1.0,
        notes TEXT
    );

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
        date as Datum,
        broker as Broker,
        transaction_type as Type,
        quantity as Aantal,
        price_per_share as 'Prijs/stuk',
        currency as Valuta,
        fees as Kosten,
        (quantity * price_per_share + fees) as Totaal,
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

            # Convert ex_date terug naar string voor display
            df['ex_date'] = df['ex_date'].dt.strftime('%Y-%m-%d')

    return df


def delete_dividend(dividend_id):
    """Verwijdert een dividend entry."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM dividends WHERE id = ?", (dividend_id,))
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
            'received_count': 0
        }

    # Filter alleen ontvangen dividenden
    received_df = df[df['received'] == 1]

    if received_df.empty:
        return {
            'total_bruto': 0,
            'total_tax': 0,
            'total_netto': 0,
            'count': len(df),
            'received_count': 0
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

    return {
        'total_bruto': total_bruto,
        'total_tax': total_tax,
        'total_netto': total_netto,
        'count': len(df),
        'received_count': len(received_df)
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


def insert_transaction(date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, exchange_rate=1.0, notes=""):
    """Voegt een transactie toe aan de database."""
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
        # Toon tabel zonder ID kolom
        display_df = purchases_df.drop(columns=['id'])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Statistieken
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            buy_count = len(purchases_df[purchases_df['Type'] == 'BUY'])
            st.metric("Aantal aankopen", buy_count)

        with col2:
            sell_count = len(purchases_df[purchases_df['Type'] == 'SELL'])
            st.metric("Aantal verkopen", sell_count)

        with col3:
            total_fees = purchases_df['Kosten'].sum()
            st.metric("Totale kosten", f"€{total_fees:.2f}")
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

        # Maak tabel
        div_table_data = []
        for _, row in dividends_df.iterrows():
            bruto = row['bruto_amount']
            currency = row.get('currency', 'EUR')
            tax_paid = row.get('tax_paid', 1)
            received = row.get('received', 0)

            # BELANGRIJKE REGEL: Tax kan alleen betaald zijn als dividend ontvangen is
            if not received:
                tax_paid = 0

            # Check voor handmatige tax waarden
            withheld = row.get('withheld_amount', 0) or 0
            additional_tax = row.get('additional_tax_due', 0) or 0
            net_received_manual = row.get('net_received')

            # Bereken tax intelligent
            if tax_paid:
                if withheld > 0 or additional_tax > 0 or net_received_manual:
                    tax = withheld + additional_tax
                    netto = net_received_manual if net_received_manual else (bruto - tax)
                    tax_label = 'Tax (Manual)'
                else:
                    tax_result = tax_calc.calculate_tax(bruto, ticker, broker)
                    tax = tax_result['total_tax']
                    netto = tax_result['net_amount']
                    stock_info_tax = tax_calc.get_stock_info(ticker)
                    if stock_info_tax and stock_info_tax['asset_type'] == 'REIT':
                        tax_label = 'Tax (REIT)'
                    elif stock_info_tax and stock_info_tax['country'] == 'Verenigde Staten':
                        tax_label = 'Tax (US+BE)'
                    else:
                        tax_label = 'Tax'
            else:
                tax = 0
                netto = bruto
                tax_label = 'Tax'

            div_table_data.append({
                'ID': row['id'],
                'Ex-Dividend Datum': row['ex_date'],
                'Currency': currency,
                'Bruto': format_currency(bruto, currency),
                tax_label: format_currency(tax, currency) if tax_paid else '-',
                'Netto': format_currency(netto, currency),
                'Tax betaald': "✅" if tax_paid else "❌",
                'Ontvangen': "✅" if received else "❌",
                'Notities': row['notes'] if row['notes'] else '-'
            })

        div_display_df = pd.DataFrame(div_table_data)

        # Bepaal welke tax kolom te tonen
        tax_column = next((col for col in div_display_df.columns if 'Tax' in col and col not in ['Tax betaald']), 'Tax')

        # Toon tabel (zonder ID kolom in display)
        st.dataframe(
            div_display_df[['Ex-Dividend Datum', 'Currency', 'Bruto', tax_column, 'Netto', 'Tax betaald', 'Ontvangen', 'Notities']],
            use_container_width=True,
            hide_index=True
        )

        # Delete functionaliteit
        with st.expander("🗑️ Dividend verwijderen"):
            delete_id = st.selectbox(
                "Selecteer dividend om te verwijderen",
                options=div_display_df['ID'].tolist(),
                format_func=lambda x: f"{div_display_df[div_display_df['ID']==x]['Ex-Dividend Datum'].values[0]} - {div_display_df[div_display_df['ID']==x]['Bruto'].values[0]}"
            )

            if st.button("🗑️ Verwijder geselecteerd dividend", type="secondary"):
                delete_dividend(delete_id)
                st.success("Dividend verwijderd!")
                st.rerun()

        # Toon totalen
        st.divider()
        st.write(f"### 📊 Ontvangen Dividenden ({div_info['received_count']}/{div_info['count']})")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Totaal Bruto (ontvangen)", f"€{div_info['total_bruto']:.2f}")

        with col2:
            st.metric("Totaal Tax Betaald", f"€{div_info['total_tax']:.2f}")

        with col3:
            st.metric("Totaal Netto (ontvangen)", f"€{div_info['total_netto']:.2f}")
    else:
        st.info("Nog geen dividenden toegevoegd.")

    # Formulier om dividend toe te voegen
    st.divider()
    st.subheader("➕ Nieuw Dividend Toevoegen")

    with st.form("dividend_form"):
        col1, col2 = st.columns(2)

        with col1:
            div_date = st.date_input("Ex-Dividend Datum")
            div_amount = st.number_input(
                "Bruto bedrag",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                help="Voer het totale bruto dividend in (voor al je aandelen)"
            )
            div_currency = st.selectbox(
                "Currency",
                options=["EUR", "USD", "GBP", "CHF", "CAD", "AUD", "JPY"],
                index=0
            )

        with col2:
            div_notes = st.text_area("Notities (optioneel)", height=40)
            div_received = st.checkbox(
                "✅ Ontvangen",
                value=False,
                help="Vink aan als je dit dividend al ontvangen hebt"
            )
            div_tax_paid = st.checkbox(
                "💰 Tax betaald",
                value=False,
                help="Vink aan als je belasting betaald hebt (kan alleen als dividend ontvangen is)"
            )

        st.info("ℹ️ Tax kan alleen betaald zijn als dividend ontvangen is")

        div_submitted = st.form_submit_button("💾 Dividend opslaan", type="primary")

        if div_submitted and div_amount > 0:
            final_tax_paid = div_tax_paid and div_received

            if div_tax_paid and not div_received:
                st.warning("⚠️ Tax betaald is uitgezet omdat dividend nog niet ontvangen is")

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
            st.success(f"✓ Dividend van {format_currency(div_amount, div_currency)} toegevoegd!")
            st.rerun()

with tab3:
    st.subheader("➕ Nieuwe Transactie Toevoegen")

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            tx_date = st.date_input("Datum")

            # Broker dropdown
            available_brokers = get_available_brokers()
            if available_brokers:
                broker_options = available_brokers + ["➕ Andere (nieuwe broker toevoegen)"]
                broker_selection = st.selectbox("Broker", broker_options)

                if broker_selection == "➕ Andere (nieuwe broker toevoegen)":
                    tx_broker = st.text_input("Nieuwe broker naam")
                else:
                    tx_broker = broker_selection
            else:
                tx_broker = st.text_input("Broker")

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
                "Kosten (broker + taksen)",
                min_value=0.0,
                step=0.01,
                format="%.2f"
            )
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
                    1.0,
                    tx_notes
                )
                st.success(f"✓ {tx_type} transactie opgeslagen!")
                st.rerun()
