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

    # Voeg fees_currency kolom toe als die nog niet bestaat
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE transactions ADD COLUMN fees_currency TEXT DEFAULT 'EUR'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Kolom bestaat al

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


def get_api_setting(key):
    """Haal een API setting op."""
    conn = get_connection()
    cursor = conn.cursor()

    # Maak tabel als die niet bestaat
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_settings (
            id INTEGER PRIMARY KEY,
            setting_key TEXT NOT NULL UNIQUE,
            setting_value TEXT,
            updated_at TEXT
        )
    """)

    cursor.execute("SELECT setting_value FROM api_settings WHERE setting_key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def get_marketstack_cache(isin):
    """Haalt gecachte Marketstack prijs op als die van vandaag is."""
    conn = get_connection()
    cursor = conn.cursor()

    # Maak marketstack_cache tabel als die niet bestaat
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS marketstack_cache (
            isin TEXT PRIMARY KEY,
            current_price REAL,
            change_percent REAL,
            currency TEXT,
            cached_date TEXT
        )
    """)

    # Haal cache op als die van vandaag is
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute(
        "SELECT current_price, change_percent, currency FROM marketstack_cache WHERE isin = ? AND cached_date = ?",
        (isin, today)
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            'current_price': result[0],
            'change_percent': result[1],
            'currency': result[2],
            'source': 'marketstack (cached)'
        }
    return None


def save_marketstack_cache(isin, current_price, change_percent, currency):
    """Slaat Marketstack prijs op in dagelijkse cache."""
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')

    cursor.execute("""
        INSERT INTO marketstack_cache (isin, current_price, change_percent, currency, cached_date)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(isin) DO UPDATE SET
            current_price = ?, change_percent = ?, currency = ?, cached_date = ?
    """, (isin, current_price, change_percent, currency, today,
          current_price, change_percent, currency, today))

    conn.commit()
    conn.close()


def get_manual_price(ticker):
    """Haalt handmatig ingestelde prijs op voor een ticker."""
    conn = get_connection()
    cursor = conn.cursor()

    # Maak tabel als die niet bestaat
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS manual_prices (
            ticker TEXT PRIMARY KEY,
            price REAL NOT NULL,
            currency TEXT DEFAULT 'EUR',
            updated_at TEXT
        )
    """)

    cursor.execute("SELECT price, currency, updated_at FROM manual_prices WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            'current_price': result[0],
            'change_percent': 0,
            'currency': result[1],
            'source': 'manual'
        }
    return None


def get_price_from_marketstack(isin):
    """
    Haalt prijs op via Marketstack API op basis van ISIN.
    Gebruikt dagelijkse cache om API calls te beperken.
    Retourneert dict met current_price, change_percent, currency of None bij fout.
    """
    if not isin:
        return None

    # Check eerst dagelijkse cache
    cached = get_marketstack_cache(isin)
    if cached:
        return cached

    api_key = get_api_setting('marketstack_api_key')
    if not api_key:
        return None

    try:
        # Marketstack endpoint voor ISIN lookup
        url = f"http://api.marketstack.com/v1/eod/latest"
        params = {
            'access_key': api_key,
            'symbols': isin
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        if 'data' in data and len(data['data']) > 0:
            stock_data = data['data'][0]
            current_price = stock_data.get('close', 0)
            open_price = stock_data.get('open', current_price)

            # Bereken change percent
            change_percent = ((current_price - open_price) / open_price * 100) if open_price > 0 else 0

            # Sla op in dagelijkse cache
            save_marketstack_cache(isin, current_price, change_percent, 'EUR')

            return {
                'current_price': float(current_price),
                'change_percent': float(change_percent),
                'currency': 'EUR',
                'source': 'marketstack'
            }

        return None

    except Exception as e:
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


def get_current_exchange_rate(from_currency='USD', to_currency='EUR'):
    """
    Haalt de huidige wisselkoers op via yfinance.
    Retourneert de koers om from_currency naar to_currency te converteren.
    """
    if from_currency == to_currency:
        return 1.0

    try:
        # yfinance format: EURUSD=X betekent 1 EUR = X USD
        # We willen USDEUR, dus EURUSD=X en dan 1/rate
        ticker_symbol = f"{from_currency}{to_currency}=X"
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1d")

        if not data.empty:
            return data['Close'].iloc[-1]

        # Fallback: probeer inverse
        ticker_symbol = f"{to_currency}{from_currency}=X"
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1d")

        if not data.empty:
            return 1 / data['Close'].iloc[-1]

        return 1.0  # Fallback

    except Exception as e:
        st.warning(f"⚠️ Kon wisselkoers niet ophalen: {e}")
        return 1.0  # Fallback naar 1:1


def convert_to_eur(amount, currency, exchange_rate=None):
    """
    Converteert een bedrag naar EUR.
    Als exchange_rate gegeven is, gebruik die. Anders haal huidige koers op.
    """
    if currency == 'EUR':
        return amount

    if exchange_rate and exchange_rate > 0:
        return amount * exchange_rate

    # Haal huidige wisselkoers op
    rate = get_current_exchange_rate(currency, 'EUR')
    return amount * rate


def get_yahoo_ticker_override(ticker):
    """Haalt de handmatige Yahoo ticker override op voor een ticker."""
    conn = get_connection()
    cursor = conn.cursor()

    # Voeg yahoo_ticker kolom toe als die nog niet bestaat
    try:
        cursor.execute("ALTER TABLE stock_info ADD COLUMN yahoo_ticker TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Kolom bestaat al

    cursor.execute("SELECT yahoo_ticker FROM stock_info WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result and result[0] else None


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

        if not data or len(data) == 0:
            return {"error": "Geen resultaten gevonden"}

        first_result = data[0]

        if 'error' in first_result:
            return {"error": f"ISIN niet gevonden: {first_result.get('error', 'Unknown error')}"}

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
    # Voor EUR berekening: gebruik gewogen gemiddelde exchange_rate voor aankopen
    # Fees: als fees_currency = EUR, gebruik direct; anders converteer met exchange_rate
    query = """
    SELECT
        ticker,
        isin,
        name,
        transaction_type,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as total_quantity,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price_per_share ELSE 0 END) as total_invested,
        SUM(CASE WHEN transaction_type = 'BUY' THEN fees + COALESCE(taxes, 0) ELSE 0 END) as total_fees,
        currency,
        MIN(date) as first_purchase_date,
        (SELECT broker FROM transactions t2 WHERE t2.ticker = transactions.ticker ORDER BY date DESC LIMIT 1) as broker,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price_per_share * COALESCE(exchange_rate, 1.0) ELSE 0 END) as total_invested_eur,
        SUM(CASE WHEN transaction_type = 'BUY' THEN
            CASE WHEN COALESCE(fees_currency, 'EUR') = 'EUR'
                THEN fees + COALESCE(taxes, 0)
                ELSE (fees + COALESCE(taxes, 0)) * COALESCE(exchange_rate, 1.0)
            END
        ELSE 0 END) as total_fees_eur,
        AVG(CASE WHEN transaction_type = 'BUY' THEN COALESCE(exchange_rate, 1.0) END) as avg_exchange_rate
    FROM transactions
    GROUP BY ticker, isin, name, currency
    HAVING total_quantity > 0
    ORDER BY name
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if not df.empty:
        # Bereken gemiddelde aankoopprijs per aandeel (inclusief fees) in originele valuta
        df['avg_purchase_price'] = (df['total_invested'] + df['total_fees']) / df['total_quantity']
        df['total_invested_with_fees'] = df['total_invested'] + df['total_fees']

        # Bereken ook in EUR (voor niet-EUR aandelen)
        df['avg_purchase_price_eur'] = (df['total_invested_eur'] + df['total_fees_eur']) / df['total_quantity']
        df['total_invested_with_fees_eur'] = df['total_invested_eur'] + df['total_fees_eur']

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
            # Map exchange codes naar Yahoo Finance suffixes
            exchange_suffix_map = {
                'BB': '.BR',  # Brussels
                'AS': '.AS',  # Amsterdam
                'PA': '.PA',  # Paris
                'MI': '.MI',  # Milan
                'MC': '.MC',  # Madrid
                'LS': '.L',   # London
                'LN': '.L',   # London
                'SW': '.SW',  # Swiss
                'GY': '.DE',  # Germany (Xetra)
                'GR': '.DE',  # Germany
                'GF': '.F',   # Frankfurt
                'GD': '.DU',  # Dusseldorf
                'US': '',     # US - geen suffix nodig
                'UA': '',     # US ARCA
                'UC': '',     # US Chicago
                'UN': '',     # US NYSE
                'UP': '',     # US Pink Sheets
                'UQ': '',     # US NASDAQ
            }

            # Bepaal prioriteit op basis van ISIN
            # Voor US ISINs: US exchanges eerst
            # Voor andere ISINs: Europese exchanges eerst
            if isin and isin.startswith('US'):
                # US ISIN: gebruik US exchanges
                priority_exchanges = ['US', 'UN', 'UQ', 'UA', 'UP', 'UC']
            else:
                # Europese ISIN: gebruik Europese exchanges, vermijd US
                priority_exchanges = ['AS', 'BB', 'LN', 'LS', 'GY', 'GR', 'GF', 'PA', 'MI', 'MC', 'SW', 'GD']

            # Eerst proberen: exchanges volgens prioriteit
            for exchange_code in priority_exchanges:
                for security in first_result['data']:
                    if security.get('exchCode', '') == exchange_code:
                        ticker = security.get('ticker', '')
                        suffix = exchange_suffix_map.get(exchange_code, '')
                        if ticker:
                            return f"{ticker}{suffix}"

            # Fallback: pak de eerste beschikbare
            for security in first_result['data']:
                ticker = security.get('ticker', '')
                exchange_code = security.get('exchCode', '')
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

    # Check eerst voor handmatige prijs (hoogste prioriteit)
    manual = get_manual_price(ticker)
    if manual:
        return manual

    # Check cache als refresh niet expliciet gevraagd is
    if not refresh:
        cached = get_cached_price(ticker)
        if cached:
            return cached

    # Voor Europese ISINs (niet-US), probeer eerst Marketstack
    # Dit geeft betere resultaten voor Europese beurzen
    if isin and not isin.startswith('US'):
        marketstack_result = get_price_from_marketstack(isin)
        if marketstack_result:
            save_price_to_cache(ticker, marketstack_result['current_price'],
                               marketstack_result['change_percent'], marketstack_result['currency'])
            return marketstack_result

    # Probeer de juiste ticker te krijgen via ISIN voor Yahoo
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

            # Sla op in cache met de ORIGINELE ticker (niet yahoo_ticker)
            # Zo kunnen we het later terugvinden met dezelfde ticker
            save_price_to_cache(ticker, result['current_price'], result['change_percent'], result['currency'])

            return result

        # Fallback naar info als history niet werkt
        info = stock.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')

        if current_price is None:
            # Probeer Marketstack als Yahoo geen data heeft
            if isin:
                marketstack_result = get_price_from_marketstack(isin)
                if marketstack_result:
                    save_price_to_cache(ticker, marketstack_result['current_price'],
                                       marketstack_result['change_percent'], marketstack_result['currency'])
                    return marketstack_result
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
        # Bij error, probeer Marketstack als fallback
        if isin:
            marketstack_result = get_price_from_marketstack(isin)
            if marketstack_result:
                save_price_to_cache(ticker, marketstack_result['current_price'],
                                   marketstack_result['change_percent'], marketstack_result['currency'])
                return marketstack_result

        # Als Marketstack ook niet werkt, probeer cache
        cached = get_cached_price(ticker)
        if cached:
            return cached
        return {"error": f"Fout bij ophalen prijs: {str(e)}"}


def calculate_performance(avg_purchase_price_eur, current_price, quantity, ticker=None, currency='EUR', current_exchange_rate=1.0):
    """
    Berekent winst/verlies statistieken, inclusief ontvangen dividenden.
    Alle berekeningen in EUR.

    avg_purchase_price_eur: gemiddelde aankoopprijs reeds in EUR
    current_price: huidige prijs in originele valuta
    current_exchange_rate: wisselkoers voor conversie naar EUR (voor USD aandelen)
    """
    total_invested = avg_purchase_price_eur * quantity

    # Converteer huidige prijs naar EUR
    if currency != 'EUR' and current_exchange_rate:
        current_price_eur = current_price * current_exchange_rate
    else:
        current_price_eur = current_price

    current_value = current_price_eur * quantity

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
        'current_price_eur': current_price_eur,
        'total_dividends_netto': total_dividends_netto,
        'total_gain_loss': total_gain_loss,
        'total_gain_loss_excl_div': current_value - total_invested,
        'gain_loss_percent': gain_loss_percent
    }


# --------- UI ---------
# Custom CSS voor compactere rijen
st.markdown("""
<style>
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
    }
    div[data-testid="column"] button {
        padding: 0.25rem 0.5rem !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="column"] p {
        margin-bottom: 0 !important;
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar voor filters
with st.sidebar:
    st.header("🔍 Filters")

    # Haal portfolio data op voor filters
    holdings = get_portfolio_holdings()

    if not holdings.empty:
        # Broker filter
        all_brokers = holdings['broker'].unique().tolist()
        broker_filter = st.multiselect("Broker", options=all_brokers, default=all_brokers)

        # Naam filter
        name_filter = st.text_input("Zoek op naam/ticker", "")

        # Sorteer opties
        st.divider()
        st.subheader("📊 Sortering")
        sort_column = st.selectbox(
            "Sorteer op",
            ["Aandeel", "Broker", "Aantal", "Aankoop", "Huidig", "Geïnvesteerd", "W/V", "Dividend"],
            index=0
        )
        sort_order = st.radio("Volgorde", ["Oplopend", "Aflopend"], horizontal=True)
    else:
        broker_filter = []
        name_filter = ""
        sort_column = "Aandeel"
        sort_order = "Oplopend"

st.title("📊 Portfolio Overzicht")

# Check if we need to show detail view
if 'view_detail_ticker' in st.query_params:
    # Redirect to detail view
    st.session_state['selected_ticker'] = st.query_params['view_detail_ticker']
    st.switch_page("pages/1_Portfolio_Detail.py")

# Haal portfolio data op
holdings = get_portfolio_holdings()

# Knop voor nieuw aandeel
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
with col_btn1:
    if st.button("➕ Nieuw Aandeel", type="primary", use_container_width=True):
        st.session_state['show_new_stock_form'] = True
        st.rerun()

with col_btn2:
    if st.button("🔄 Ververs Koersen", type="secondary", use_container_width=True):
        st.session_state['refresh_prices'] = True
        st.rerun()

# Toon laatste update tijd
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

            st.info(f"📅 Laatste koers update: {last_update_dt.strftime('%d/%m/%Y %H:%M')} ({time_str})")
        except:
            st.info("📅 Laatste koers update: Onbekend")
    else:
        st.info("📅 Koersen nog niet opgehaald")

refresh_prices = st.session_state.get('refresh_prices', False)
if refresh_prices:
    st.session_state['refresh_prices'] = False  # Reset flag

# Toon formulier voor nieuw aandeel als gevraagd
if st.session_state.get('show_new_stock_form', False):
    with st.expander("➕ Nieuw Aandeel Toevoegen", expanded=True):
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
            st.write("")
            st.write("")
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

        # Show lookup result
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

        st.divider()

        # Transaction Form
        with st.form("new_stock_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                date = st.date_input("Datum", format="DD/MM/YYYY")

                # Broker dropdown
                available_brokers = get_available_brokers()
                if available_brokers:
                    broker = st.selectbox("Broker", available_brokers)
                else:
                    st.error("⚠️ Geen brokers geconfigureerd. Ga naar **Broker Settings**")
                    broker = None

                transaction_type = st.selectbox("Type", ["BUY", "SELL"])
                name = st.text_input("Naam aandeel", value=st.session_state.security_data.get('name', ''))

            with col2:
                ticker = st.text_input("Ticker (bv. AAPL)", value=st.session_state.security_data.get('ticker', ''))
                isin = st.text_input("ISIN", value=st.session_state.security_data.get('isin', ''))
                quantity = st.number_input("Aantal aandelen", min_value=1, step=1)
                price_per_share = st.number_input("Prijs per aandeel", min_value=0.0, step=0.01, format="%.2f")

            with col3:
                default_currency = st.session_state.security_data.get('currency', 'EUR')
                currency_options = ["EUR", "USD", "GBP"]
                if default_currency not in currency_options:
                    default_currency = "EUR"

                currency = st.selectbox("Valuta", currency_options, index=currency_options.index(default_currency))
                fees = st.number_input("Kosten (broker + taksen)", min_value=0.0, step=0.01, format="%.2f")
                exchange_rate = st.number_input("Wisselkoers naar EUR", min_value=0.0, value=1.0, step=0.01, format="%.4f")
                notes = st.text_area("Notities (optioneel)", height=60)

            col_submit, col_cancel = st.columns(2)
            with col_submit:
                submitted = st.form_submit_button("💾 Opslaan", type="primary", use_container_width=True)
            with col_cancel:
                cancel = st.form_submit_button("✗ Annuleer", use_container_width=True)

            if cancel:
                st.session_state['show_new_stock_form'] = False
                st.session_state.security_data = {}
                st.session_state.lookup_result = None
                st.rerun()

            if submitted:
                if not (broker and name and ticker and isin and quantity > 0):
                    st.error("Vul alle verplichte velden in.")
                else:
                    insert_transaction(date, broker, transaction_type, name, ticker, isin, quantity, price_per_share, currency, fees, exchange_rate, notes)
                    st.success("✓ Transactie opgeslagen.")
                    st.session_state['show_new_stock_form'] = False
                    st.session_state.security_data = {}
                    st.session_state.lookup_result = None
                    st.rerun()

st.divider()

# Bereken totalen
portfolio_data = []

spinner_text = "Koersen verversen..." if refresh_prices else "Huidige prijzen ophalen..."

# Haal wisselkoersen op voor niet-EUR valuta's
exchange_rates = {}
currencies_needed = holdings['currency'].unique() if not holdings.empty else []
for curr in currencies_needed:
    if curr != 'EUR':
        exchange_rates[curr] = get_current_exchange_rate(curr, 'EUR')

with st.spinner(spinner_text):
    for idx, row in holdings.iterrows():
        ticker = row['ticker']
        isin = row['isin']
        broker = row['broker']
        currency = row['currency']

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

            # Gebruik de gemiddelde wisselkoers van de aankopen
            # Als exchange_rate = 1.0 voor USD, dan geen conversie (USD rekening)
            avg_exchange_rate = row.get('avg_exchange_rate', 1.0)

            # Als de originele transactie geen conversie had (rate = 1.0),
            # gebruik dan ook 1.0 voor huidige prijs (blijf in originele valuta)
            if currency != 'EUR' and avg_exchange_rate == 1.0:
                current_exchange_rate = 1.0  # Geen conversie - USD rekening
            else:
                current_exchange_rate = exchange_rates.get(currency, 1.0)

            # Bereken performance (inclusief dividenden)
            perf = calculate_performance(
                row['avg_purchase_price_eur'],
                current_price,
                row['total_quantity'],
                ticker,
                currency,
                current_exchange_rate
            )

            # Haal dividend info op
            div_info = calculate_total_dividends(ticker)

            # Bepaal valutasymbool voor display
            is_usd_account = (currency == 'USD' and avg_exchange_rate == 1.0)
            display_symbol = '$' if is_usd_account else '€'

            # Bereken W/V in EUR voor USD aandelen (met huidige wisselkoers)
            actual_eur_rate = exchange_rates.get(currency, 1.0) if currency != 'EUR' else 1.0
            wv_in_eur = perf['total_gain_loss'] * actual_eur_rate if is_usd_account else perf['total_gain_loss']
            invested_in_eur = perf['total_invested'] * actual_eur_rate if is_usd_account else perf['total_invested']

            portfolio_data.append({
                'Aandeel': row['name'],
                'Ticker': ticker,
                'Broker': broker,
                'ISIN': row['isin'],
                'Aantal': int(row['total_quantity']),
                'Avg Aankoopprijs': f"{display_symbol}{row['avg_purchase_price_eur']:.2f}",
                'Huidige Prijs': f"{display_symbol}{perf['current_price_eur']:.2f}",
                'Totaal Geinvesteerd': f"{display_symbol}{perf['total_invested']:.2f}",
                'Huidige Waarde': f"{display_symbol}{perf['current_value']:.2f}",
                'Dividend (netto)': f"€{div_info['total_netto']:.2f}",
                'W/V (excl. div)': f"{display_symbol}{perf['total_gain_loss_excl_div']:.2f}",
                'W/V (incl. div)': f"{display_symbol}{perf['total_gain_loss']:.2f}",
                'Performance': f"{perf['gain_loss_percent']:+.2f}%",
                '_current_price': current_price,
                '_perf': perf,
                '_avg_price': row['avg_purchase_price_eur'],
                '_quantity': row['total_quantity'],
                '_currency': currency,
                '_is_usd_account': is_usd_account,
                '_wv_in_eur': wv_in_eur,
                '_invested_in_eur': invested_in_eur,
                '_display_symbol': display_symbol
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
                'Avg Aankoopprijs': f"€{row['avg_purchase_price_eur']:.2f}",
                'Huidige Prijs': "N/A",
                'Totaal Geinvesteerd': f"€{row['total_invested_with_fees_eur']:.2f}",
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
    # Bereken totalen alleen als er data is (alles in EUR)
    if portfolio_data:
        total_invested = 0
        total_current = 0

        for row in portfolio_data:
            if '_perf' not in row or not isinstance(row['_perf'], dict):
                continue

            if row.get('_is_usd_account'):
                # USD rekening: converteer naar EUR met huidige wisselkoers
                total_invested += row.get('_invested_in_eur', row['_perf']['total_invested'])
                # current_value in USD * EUR/USD rate
                currency = row.get('_currency', 'EUR')
                eur_rate = exchange_rates.get(currency, 1.0)
                total_current += row['_perf']['current_value'] * eur_rate
            else:
                total_invested += row['_perf']['total_invested']
                total_current += row['_perf']['current_value']
        total_dividends = sum([row['_perf']['total_dividends_netto'] for row in portfolio_data if '_perf' in row and isinstance(row['_perf'], dict)])
        total_gain_loss_excl = total_current - total_invested
        total_gain_loss_incl = total_gain_loss_excl + total_dividends
        total_gain_loss_percent = (total_gain_loss_incl / total_invested * 100) if total_invested > 0 else 0

        # Toon totalen BOVENAAN
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

        # Debug sectie
        with st.expander("🔍 Debug: Wisselkoersen & Prijzen"):
            st.write("### Gebruikte Wisselkoersen")
            for curr, rate in exchange_rates.items():
                st.write(f"**{curr}/EUR:** {rate:.4f}")

            st.divider()
            st.write("### Waarde per Aandeel")

            # Tabel met debug info
            debug_data = []
            for row in portfolio_data:
                if '_perf' in row and isinstance(row['_perf'], dict):
                    currency = row.get('_currency', 'EUR')
                    original_price = row.get('_current_price', 0)
                    eur_price = row['_perf'].get('current_price_eur', original_price)
                    quantity = row.get('_quantity', 0)
                    eur_value = row['_perf'].get('current_value', 0)
                    rate_used = exchange_rates.get(currency, 1.0) if currency != 'EUR' else 1.0

                    # Haal Yahoo ticker op uit cache
                    ticker = row['Ticker']
                    isin = row.get('ISIN', '')
                    yahoo_ticker = get_yahoo_ticker_from_isin(isin) if isin else ticker

                    debug_data.append({
                        'Ticker': ticker,
                        'Yahoo': yahoo_ticker or ticker,
                        'Valuta': currency,
                        'Prijs': f"{original_price:.2f}",
                        'Koers': f"{rate_used:.4f}" if currency != 'EUR' else "-",
                        'EUR': f"€{eur_price:.2f}",
                        'Qty': int(quantity),
                        'Totaal': f"€{eur_value:.2f}"
                    })

            if debug_data:
                debug_df = pd.DataFrame(debug_data)
                st.dataframe(debug_df, use_container_width=True, hide_index=True)

                # Totaal check
                total_from_table = sum([row['_perf']['current_value'] for row in portfolio_data if '_perf' in row and isinstance(row['_perf'], dict)])
                st.write(f"**Som van alle aandelen:** €{total_from_table:.2f}")
                st.write(f"**Getoonde Huidige Waarde:** €{total_current:.2f}")

        st.divider()

    # Sorteer portfolio_df
    ascending = sort_order == "Oplopend"

    # Map display column naar sorteerbare waarde
    if sort_column == "Aandeel":
        portfolio_df = portfolio_df.sort_values(by="Aandeel", ascending=ascending)
    elif sort_column == "Broker":
        portfolio_df = portfolio_df.sort_values(by="Broker", ascending=ascending)
    elif sort_column == "Aantal":
        portfolio_df = portfolio_df.sort_values(by="Aantal", ascending=ascending)
    elif sort_column == "Aankoop":
        portfolio_df['_sort_avg'] = portfolio_df.apply(
            lambda row: row['_avg_price'] if '_avg_price' in row else 0, axis=1
        )
        portfolio_df = portfolio_df.sort_values(by="_sort_avg", ascending=ascending)
        portfolio_df = portfolio_df.drop(columns=['_sort_avg'])
    elif sort_column == "Huidig":
        portfolio_df['_sort_current'] = portfolio_df.apply(
            lambda row: row['_current_price'] if '_current_price' in row else 0, axis=1
        )
        portfolio_df = portfolio_df.sort_values(by="_sort_current", ascending=ascending)
        portfolio_df = portfolio_df.drop(columns=['_sort_current'])
    elif sort_column == "Geïnvesteerd":
        # Sorteer op numerieke waarde
        portfolio_df['_sort_invested'] = portfolio_df.apply(
            lambda row: row['_perf']['total_invested'] if '_perf' in row and isinstance(row['_perf'], dict) else 0, axis=1
        )
        portfolio_df = portfolio_df.sort_values(by="_sort_invested", ascending=ascending)
        portfolio_df = portfolio_df.drop(columns=['_sort_invested'])
    elif sort_column == "W/V":
        portfolio_df['_sort_wv'] = portfolio_df.apply(
            lambda row: row['_perf']['total_gain_loss'] if '_perf' in row and isinstance(row['_perf'], dict) else 0, axis=1
        )
        portfolio_df = portfolio_df.sort_values(by="_sort_wv", ascending=ascending)
        portfolio_df = portfolio_df.drop(columns=['_sort_wv'])
    elif sort_column == "Dividend":
        portfolio_df['_sort_div'] = portfolio_df.apply(
            lambda row: row['_perf']['total_dividends_netto'] if '_perf' in row and isinstance(row['_perf'], dict) else 0, axis=1
        )
        portfolio_df = portfolio_df.sort_values(by="_sort_div", ascending=ascending)
        portfolio_df = portfolio_df.drop(columns=['_sort_div'])

    # Header rij
    header_col1, header_col2, header_col3, header_col4, header_col5, header_col6, header_col7, header_col8 = st.columns([2, 0.7, 0.8, 0.8, 1, 0.9, 1.2, 0.8])
    with header_col1:
        st.markdown("**Aandeel**")
    with header_col2:
        st.markdown("**Aantal**")
    with header_col3:
        st.markdown("**Aankoop**")
    with header_col4:
        st.markdown("**Huidig**")
    with header_col5:
        st.markdown("**Geïnvesteerd**")
    with header_col6:
        st.markdown("**W/V**")
    with header_col7:
        st.markdown("**Dividend**")
    with header_col8:
        st.markdown("**Broker**")

    st.divider()

    # Toon portfolio als compacte lijst met hyperlinks
    for idx, row in portfolio_df.iterrows():
        # Bepaal kleur voor performance
        if '_perf' in row and isinstance(row['_perf'], dict):
            perf_value = row['_perf']['total_gain_loss']
            perf_color = "green" if perf_value >= 0 else "red"
        else:
            perf_color = "gray"
            perf_value = 0

        # Bepaal currency symbool
        currency_symbol = "$" if 'ISIN' in row and row['ISIN'].startswith('US') else "€"

        # Container voor elke rij
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 0.7, 0.8, 0.8, 1, 0.9, 1.2, 0.8])

        with col1:
            # Link-stijl met klein icoon
            subcols = st.columns([0.5, 4])
            with subcols[0]:
                if st.button("▶", key=f"go_{row['Ticker']}", help="Details"):
                    st.session_state['selected_ticker'] = row['Ticker']
                    st.switch_page("pages/1_Portfolio_Detail.py")
            with subcols[1]:
                st.markdown(f"<span style='color: #0066cc; font-size: 0.9rem;'>{row['Aandeel']}</span> <span style='color: #666; font-size: 0.85rem;'>({row['Ticker']})</span>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<small>{row['Aantal']}</small>", unsafe_allow_html=True)

        with col3:
            if '_avg_price' in row:
                st.markdown(f"<small>{currency_symbol}{row['_avg_price']:.2f}</small>", unsafe_allow_html=True)
            else:
                st.markdown("<small>N/A</small>", unsafe_allow_html=True)

        with col4:
            if '_current_price' in row:
                st.markdown(f"<small>{currency_symbol}{row['_current_price']:.2f}</small>", unsafe_allow_html=True)
            else:
                st.markdown("<small>N/A</small>", unsafe_allow_html=True)

        with col5:
            st.markdown(f"<small>{row['Totaal Geinvesteerd']}</small>", unsafe_allow_html=True)

        with col6:
            # Winst/Verlies
            if '_perf' in row and isinstance(row['_perf'], dict):
                wv = row['_perf']['total_gain_loss']
                display_symbol = row.get('_display_symbol', '€')
                is_usd_account = row.get('_is_usd_account', False)

                if is_usd_account:
                    # Toon zowel USD als EUR
                    wv_eur = row.get('_wv_in_eur', wv)
                    eur_color = "green" if wv_eur >= 0 else "red"
                    st.markdown(f"<small style='color: {perf_color};'><b>{display_symbol}{wv:+.2f}</b></small><br><small style='color: {eur_color};'>€{wv_eur:+.2f}</small>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<small style='color: {perf_color};'><b>€{wv:+.2f}</b></small>", unsafe_allow_html=True)
            else:
                st.markdown("<small>N/A</small>", unsafe_allow_html=True)

        with col7:
            st.markdown(f"<small>{row['Dividend (netto)']}</small>", unsafe_allow_html=True)

        with col8:
            st.markdown(f"<small>{row['Broker']}</small>", unsafe_allow_html=True)

    st.divider()
    st.info("💡 Klik op een aandeel om de details te bekijken")
