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
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

st.set_page_config(page_title="Portfolio Overzicht", page_icon="📊", layout="wide")


# --------- Database helpers ---------
def get_connection():
    """Maakt verbinding met de database."""
    conn = sqlite3.connect(DB_PATH)

    # Zorg dat dividends tabel bestaat
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS dividends (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        isin TEXT NOT NULL,
        ex_date TEXT NOT NULL,
        bruto_amount REAL NOT NULL,
        notes TEXT
    );
    """)

    # Zorg dat price_cache tabel bestaat
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS price_cache (
        ticker TEXT PRIMARY KEY,
        current_price REAL NOT NULL,
        change_percent REAL,
        currency TEXT,
        updated_at TEXT NOT NULL
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


def get_cached_price(ticker):
    """Haalt de gecachte prijs op voor een ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT current_price, change_percent, currency, updated_at FROM price_cache WHERE ticker = ?",
        (ticker,)
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            'current_price': result[0],
            'change_percent': result[1],
            'currency': result[2],
            'updated_at': result[3]
        }
    return None


def save_price_to_cache(ticker, current_price, change_percent, currency):
    """Slaat een prijs op in de cache."""
    conn = get_connection()
    cursor = conn.cursor()
    updated_at = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO price_cache (ticker, current_price, change_percent, currency, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            current_price = ?,
            change_percent = ?,
            currency = ?,
            updated_at = ?
        """,
        (ticker, current_price, change_percent, currency, updated_at,
         current_price, change_percent, currency, updated_at)
    )
    conn.commit()
    conn.close()


def get_last_update_time():
    """Haalt de laatste update tijd op van alle gecachte prijzen."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(updated_at) FROM price_cache")
    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        return result[0]
    return None


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


def get_manual_dividends(ticker):
    """Haalt alle handmatig ingevoerde dividenden op voor een ticker."""
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
    return df


def delete_dividend(dividend_id):
    """Verwijdert een dividend entry."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM dividends WHERE id = ?", (dividend_id,))
    conn.commit()
    conn.close()


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


def calculate_total_dividends(ticker, filter_ownership=True):
    """Berekent totaal ontvangen dividend (bruto en netto) voor een ticker - ALLEEN ontvangen dividenden."""
    df = get_manual_dividends(ticker)

    if not df.empty and filter_ownership:
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


def get_portfolio_holdings():
    """
    Haalt alle holdings op gegroepeerd per aandeel.
    Berekent totaal aantal aandelen, gemiddelde aankoopprijs per aandeel.
    Inclusief broker informatie (meest recente broker per ticker).
    """
    conn = get_connection()

    # Haal alle transacties op met broker info
    query = """
    SELECT
        ticker,
        isin,
        name,
        transaction_type,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as total_quantity,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price_per_share ELSE 0 END) as total_invested,
        SUM(CASE WHEN transaction_type = 'BUY' THEN fees ELSE 0 END) as total_fees,
        currency,
        MIN(date) as first_purchase_date,
        (SELECT broker FROM transactions t2 WHERE t2.ticker = transactions.ticker ORDER BY date DESC LIMIT 1) as broker
    FROM transactions
    GROUP BY ticker, isin, name, currency
    HAVING total_quantity > 0
    ORDER BY name
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if not df.empty:
        # Bereken gemiddelde aankoopprijs per aandeel (inclusief fees)
        df['avg_purchase_price'] = (df['total_invested'] + df['total_fees']) / df['total_quantity']
        df['total_invested_with_fees'] = df['total_invested'] + df['total_fees']

    return df


def get_yahoo_ticker_from_isin(isin):
    """
    Haalt de correcte Yahoo Finance ticker op via ISIN lookup in OpenFIGI.
    Retourneert ticker met exchange suffix (bv. KBC.BR voor Brussels).
    """
    if not isin:
        return None

    try:
        headers = {'Content-Type': 'application/json'}
        jobs = [{'idType': 'ID_ISIN', 'idValue': isin.strip().upper()}]

        response = requests.post(
            url=OPENFIGI_URL,
            headers=headers,
            json=jobs,
            timeout=10
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if not data or len(data) == 0:
            return None

        first_result = data[0]

        if 'data' in first_result and len(first_result['data']) > 0:
            # Zoek naar een match met compositeFIGI (meest betrouwbaar voor yahoo)
            for security in first_result['data']:
                ticker = security.get('ticker', '')
                exchange_code = security.get('exchCode', '')

                # Map exchange codes naar Yahoo Finance suffixes
                exchange_suffix_map = {
                    'BB': '.BR',  # Brussels
                    'AS': '.AS',  # Amsterdam
                    'PA': '.PA',  # Paris
                    'MI': '.MI',  # Milan
                    'MC': '.MC',  # Madrid
                    'LS': '.L',   # London
                    'SW': '.SW',  # Swiss
                    'GY': '.DE',  # Germany (Xetra)
                    'GR': '.DE',  # Germany
                    'US': '',     # US - geen suffix nodig
                }

                suffix = exchange_suffix_map.get(exchange_code, '')
                if ticker:
                    return f"{ticker}{suffix}"

        return None

    except Exception:
        # Silently fail - we'll fall back to the original ticker
        return None


def get_current_price(ticker, isin=None, refresh=False):
    """
    Haalt de huidige prijs op voor een ticker via yfinance API.
    Als ISIN gegeven is, wordt eerst de juiste Yahoo ticker bepaald.
    Retourneert dict met current_price, change_percent, currency.

    Als refresh=False, wordt eerst in de cache gekeken.
    Als refresh=True, wordt altijd de API bevraagd en cache geupdate.
    """
    if not ticker:
        return None

    # Check cache als refresh niet expliciet gevraagd is
    if not refresh:
        cached = get_cached_price(ticker)
        if cached:
            return cached

    # Probeer eerst de juiste ticker te krijgen via ISIN als deze beschikbaar is
    yahoo_ticker = ticker
    if isin:
        isin_ticker = get_yahoo_ticker_from_isin(isin)
        if isin_ticker:
            yahoo_ticker = isin_ticker

    try:
        stock = yf.Ticker(yahoo_ticker)

        # Probeer eerst via history (meest betrouwbaar)
        hist = stock.history(period='5d')
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]

            # Bereken dagelijkse verandering
            if len(hist) > 1:
                prev_close = hist['Close'].iloc[-2]
                change_percent = ((current_price - prev_close) / prev_close) * 100
            else:
                change_percent = 0

            result = {
                'current_price': float(current_price),
                'change_percent': float(change_percent),
                'currency': 'EUR',  # Default to EUR for European stocks
                'name': ticker
            }

            # Sla op in cache
            save_price_to_cache(ticker, result['current_price'], result['change_percent'], result['currency'])

            return result

        # Fallback naar info als history niet werkt
        info = stock.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')

        if current_price is None:
            return {"error": "Geen prijs beschikbaar"}

        change_percent = info.get('regularMarketChangePercent', 0)
        currency = info.get('currency', 'EUR')

        result = {
            'current_price': float(current_price),
            'change_percent': float(change_percent) if change_percent else 0,
            'currency': currency,
            'name': info.get('longName', info.get('shortName', ticker))
        }

        # Sla op in cache
        save_price_to_cache(ticker, result['current_price'], result['change_percent'], result['currency'])

        return result

    except Exception as e:
        # Bij error, probeer cache als fallback
        cached = get_cached_price(ticker)
        if cached:
            return cached
        return {"error": f"Fout bij ophalen prijs: {str(e)}"}


def calculate_performance(avg_purchase_price, current_price, quantity, ticker=None):
    """Berekent winst/verlies statistieken, inclusief ontvangen dividenden."""
    total_invested = avg_purchase_price * quantity
    current_value = current_price * quantity

    # Haal dividend data op als ticker gegeven is
    total_dividends_netto = 0
    if ticker:
        div_info = calculate_total_dividends(ticker)
        total_dividends_netto = div_info['total_netto']

    # Bereken totale winst/verlies inclusief dividenden
    total_gain_loss = (current_value + total_dividends_netto) - total_invested
    gain_loss_percent = (total_gain_loss / total_invested) * 100 if total_invested > 0 else 0

    return {
        'total_invested': total_invested,
        'current_value': current_value,
        'total_dividends_netto': total_dividends_netto,
        'total_gain_loss': total_gain_loss,
        'total_gain_loss_excl_div': current_value - total_invested,
        'gain_loss_percent': gain_loss_percent
    }


# --------- UI ---------
st.title("📊 Portfolio Overzicht")

# Haal portfolio data op
holdings = get_portfolio_holdings()

if holdings.empty:
    st.info("Je portfolio is nog leeg. Voeg eerst transacties toe op de hoofdpagina.")
    st.stop()

# Check if we need to show detail view
if 'view_detail_ticker' in st.query_params:
    # Redirect to detail view
    st.session_state['selected_ticker'] = st.query_params['view_detail_ticker']
    st.switch_page("pages/1_Portfolio_Detail.py")

# Toon laatste update tijd en refresh knop
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    last_update = get_last_update_time()
    if last_update:
        try:
            last_update_dt = datetime.fromisoformat(last_update)
            time_ago = datetime.now() - last_update_dt

            if time_ago.days > 0:
                time_str = f"{time_ago.days} dag(en) geleden"
            elif time_ago.seconds > 3600:
                time_str = f"{time_ago.seconds // 3600} uur geleden"
            else:
                time_str = f"{time_ago.seconds // 60} minuten geleden"

            st.info(f"📅 Laatste koers update: {last_update_dt.strftime('%d-%m-%Y %H:%M')} ({time_str})")
        except:
            st.info("📅 Laatste koers update: Onbekend")
    else:
        st.info("📅 Koersen nog niet opgehaald")

with col2:
    st.write("")  # Spacer

with col3:
    st.write("")  # Spacer
    st.write("")  # Spacer
    if st.button("🔄 Ververs Koersen", type="primary", use_container_width=True):
        st.session_state['refresh_prices'] = True
        st.rerun()

refresh_prices = st.session_state.get('refresh_prices', False)
if refresh_prices:
    st.session_state['refresh_prices'] = False  # Reset flag

# Filters
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    # Broker filter
    all_brokers = holdings['broker'].unique().tolist()
    broker_filter = st.multiselect("Filter op Broker", options=all_brokers, default=all_brokers)

with col2:
    # Naam filter
    name_filter = st.text_input("Zoek op naam/ticker", "")

with col3:
    st.write("")

# Toon portfolio overzicht
st.subheader("Jouw Holdings")

# Bereken totalen
portfolio_data = []

spinner_text = "Koersen verversen..." if refresh_prices else "Huidige prijzen ophalen..."
with st.spinner(spinner_text):
    for idx, row in holdings.iterrows():
        ticker = row['ticker']
        isin = row['isin']
        broker = row['broker']

        # Apply filters
        if broker not in broker_filter:
            continue
        if name_filter and name_filter.lower() not in row['name'].lower() and name_filter.lower() not in ticker.lower():
            continue

        # Haal huidige prijs op (met ISIN voor juiste ticker lookup)
        # refresh=True als de refresh knop is ingedrukt
        price_info = get_current_price(ticker, isin, refresh=refresh_prices)

        if price_info and 'error' not in price_info:
            current_price = price_info['current_price']

            # Bereken performance (inclusief dividenden)
            perf = calculate_performance(
                row['avg_purchase_price'],
                current_price,
                row['total_quantity'],
                ticker
            )

            # Haal dividend info op
            div_info = calculate_total_dividends(ticker)

            portfolio_data.append({
                'Aandeel': row['name'],
                'Ticker': ticker,
                'Broker': broker,
                'ISIN': row['isin'],
                'Aantal': int(row['total_quantity']),
                'Avg Aankoopprijs': f"€{row['avg_purchase_price']:.2f}",
                'Huidige Prijs': f"€{current_price:.2f}",
                'Totaal Geinvesteerd': f"€{perf['total_invested']:.2f}",
                'Huidige Waarde': f"€{perf['current_value']:.2f}",
                'Dividend (netto)': f"€{div_info['total_netto']:.2f}",
                'W/V (excl. div)': f"€{perf['total_gain_loss_excl_div']:.2f}",
                'W/V (incl. div)': f"€{perf['total_gain_loss']:.2f}",
                'Performance': f"{perf['gain_loss_percent']:+.2f}%",
                '_current_price': current_price,
                '_perf': perf,
                '_avg_price': row['avg_purchase_price'],
                '_quantity': row['total_quantity']
            })
        else:
            # Kon prijs niet ophalen
            div_info = calculate_total_dividends(ticker)

            portfolio_data.append({
                'Aandeel': row['name'],
                'Ticker': ticker,
                'Broker': broker,
                'ISIN': row['isin'],
                'Aantal': int(row['total_quantity']),
                'Avg Aankoopprijs': f"€{row['avg_purchase_price']:.2f}",
                'Huidige Prijs': "N/A",
                'Totaal Geinvesteerd': f"€{row['total_invested_with_fees']:.2f}",
                'Huidige Waarde': "N/A",
                'Dividend (netto)': f"€{div_info['total_netto']:.2f}",
                'W/V (excl. div)': "N/A",
                'W/V (incl. div)': "N/A",
                'Performance': "N/A",
                '_error': price_info.get('error', 'Onbekende fout') if price_info else 'API error'
            })

# Maak DataFrame voor display
portfolio_df = pd.DataFrame(portfolio_data)

if portfolio_df.empty:
    st.info("Geen aandelen gevonden met de huidige filters.")
else:
    # Toon portfolio tabel (zonder hidden columns)
    display_columns = ['Aandeel', 'Ticker', 'Broker', 'Aantal', 'Avg Aankoopprijs', 'Huidige Prijs',
                       'Totaal Geinvesteerd', 'Huidige Waarde', 'Dividend (netto)', 'W/V (excl. div)', 'W/V (incl. div)', 'Performance']

    # Gebruik st.dataframe met on_select callback
    event = st.dataframe(
        portfolio_df[display_columns],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    # Handle row selection
    if len(event.selection.rows) > 0:
        selected_idx = event.selection.rows[0]
        selected_ticker = portfolio_df.iloc[selected_idx]['Ticker']
        st.query_params['view_detail_ticker'] = selected_ticker
        st.rerun()

# Bereken totalen alleen als er data is
if portfolio_data:
    total_invested = sum([row['_perf']['total_invested'] for row in portfolio_data if '_perf' in row])
    total_current = sum([row['_perf']['current_value'] for row in portfolio_data if '_perf' in row])
    total_dividends = sum([row['_perf']['total_dividends_netto'] for row in portfolio_data if '_perf' in row])
    total_gain_loss_excl = total_current - total_invested
    total_gain_loss_incl = total_gain_loss_excl + total_dividends
    total_gain_loss_percent = (total_gain_loss_incl / total_invested * 100) if total_invested > 0 else 0

    # Toon totalen
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Totaal Geinvesteerd", f"€{total_invested:,.2f}")

    with col2:
        st.metric("Huidige Waarde", f"€{total_current:,.2f}")

    with col3:
        st.metric("Totaal Dividend", f"€{total_dividends:,.2f}")

    with col4:
        st.metric(
            "W/V (excl. div)",
            f"€{total_gain_loss_excl:,.2f}",
            delta=f"{(total_gain_loss_excl/total_invested*100):+.2f}%" if total_invested > 0 else None
        )

    with col5:
        st.metric(
            "W/V (incl. div)",
            f"€{total_gain_loss_incl:,.2f}",
            delta=f"{total_gain_loss_percent:+.2f}%"
        )

    st.info("💡 Klik op een rij in de tabel om de details van een aandeel te bekijken")
