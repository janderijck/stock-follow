import streamlit as st
import sqlite3
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path
from datetime import datetime

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
        SELECT id, ex_date, bruto_amount, notes, currency, tax_paid, received
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


def calculate_total_dividends(ticker):
    """Berekent totaal ontvangen dividend (bruto en netto) voor een ticker - ALLEEN ontvangen dividenden."""
    df = get_manual_dividends(ticker)

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

    TAX_RATE = 0.30
    total_bruto = received_df['bruto_amount'].sum()
    # Only calculate tax for dividends where tax was actually paid
    total_tax = sum(row['bruto_amount'] * TAX_RATE for _, row in received_df.iterrows() if row.get('tax_paid', 1))
    total_netto = total_bruto - total_tax

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
    """
    conn = get_connection()

    # Haal alle transacties op
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
        MIN(date) as first_purchase_date
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

# Initialize session state voor geselecteerd aandeel en refresh flag
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

refresh_prices = st.session_state.get('refresh_prices', False)
if refresh_prices:
    st.session_state['refresh_prices'] = False  # Reset flag

# Toon portfolio overzicht
st.subheader("Jouw Holdings")

# Bereken totalen
portfolio_data = []

spinner_text = "Koersen verversen..." if refresh_prices else "Huidige prijzen ophalen..."
with st.spinner(spinner_text):
    for idx, row in holdings.iterrows():
        ticker = row['ticker']
        isin = row['isin']

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
                'ISIN': row['isin'],
                'Aantal': int(row['total_quantity']),
                'Avg Aankoopprijs': f"€{row['avg_purchase_price']:.2f}",
                'Huidige Prijs': f"€{current_price:.2f}",
                'Totaal Geinvesteerd': f"€{perf['total_invested']:.2f}",
                'Huidige Waarde': f"€{perf['current_value']:.2f}",
                'Dividend (netto)': f"€{div_info['total_netto']:.2f}",
                'Winst/Verlies': f"€{perf['total_gain_loss']:.2f}",
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
                'ISIN': row['isin'],
                'Aantal': int(row['total_quantity']),
                'Avg Aankoopprijs': f"€{row['avg_purchase_price']:.2f}",
                'Huidige Prijs': "N/A",
                'Totaal Geinvesteerd': f"€{row['total_invested_with_fees']:.2f}",
                'Huidige Waarde': "N/A",
                'Dividend (netto)': f"€{div_info['total_netto']:.2f}",
                'Winst/Verlies': "N/A",
                'Performance': "N/A",
                '_error': price_info.get('error', 'Onbekende fout') if price_info else 'API error'
            })

# Maak DataFrame voor display
portfolio_df = pd.DataFrame(portfolio_data)

# Toon portfolio tabel (zonder hidden columns)
display_columns = ['Aandeel', 'Ticker', 'Aantal', 'Avg Aankoopprijs', 'Huidige Prijs',
                   'Totaal Geinvesteerd', 'Huidige Waarde', 'Dividend (netto)', 'Winst/Verlies', 'Performance']

st.dataframe(
    portfolio_df[display_columns],
    use_container_width=True,
    hide_index=True
)

# Bereken totalen
total_invested = sum([row['_perf']['total_invested'] for row in portfolio_data if '_perf' in row])
total_current = sum([row['_perf']['current_value'] for row in portfolio_data if '_perf' in row])
total_gain_loss = total_current - total_invested
total_gain_loss_percent = (total_gain_loss / total_invested * 100) if total_invested > 0 else 0

# Toon totalen
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Totaal Geinvesteerd", f"€{total_invested:,.2f}")

with col2:
    st.metric("Huidige Waarde", f"€{total_current:,.2f}")

with col3:
    st.metric(
        "Totale Winst/Verlies",
        f"€{total_gain_loss:,.2f}",
        delta=f"{total_gain_loss_percent:+.2f}%"
    )

with col4:
    st.metric(
        "Performance",
        f"{total_gain_loss_percent:+.2f}%",
        delta=None
    )

# Klikbare aandeel details
st.divider()
st.subheader("Aandeel Details")

# Selecteer aandeel via selectbox
selected_ticker = st.selectbox(
    "Selecteer een aandeel voor gedetailleerde informatie:",
    options=portfolio_df['Ticker'].tolist(),
    format_func=lambda x: f"{portfolio_df[portfolio_df['Ticker']==x]['Aandeel'].values[0]} ({x})"
)

if selected_ticker:
    # Zoek geselecteerde rij
    selected_data = portfolio_df[portfolio_df['Ticker'] == selected_ticker].iloc[0]

    st.write(f"### {selected_data['Aandeel']} ({selected_ticker})")

    # Toon details in kolommen
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Basis Informatie**")
        st.write(f"- ISIN: `{selected_data['ISIN']}`")
        st.write(f"- Aantal aandelen: **{selected_data['Aantal']}**")
        st.write(f"- Avg Aankoopprijs: **{selected_data['Avg Aankoopprijs']}**")

    with col2:
        st.write("**Huidige Status**")
        st.write(f"- Huidige prijs: **{selected_data['Huidige Prijs']}**")
        st.write(f"- Totaal geinvesteerd: **{selected_data['Totaal Geinvesteerd']}**")
        st.write(f"- Huidige waarde: **{selected_data['Huidige Waarde']}**")

    with col3:
        st.write("**Performance**")
        st.write(f"- Winst/Verlies: **{selected_data['Winst/Verlies']}**")

        # Kleurcode voor performance
        if '_perf' in selected_data:
            perf_value = selected_data['_perf']['gain_loss_percent']
            color = "🟢" if perf_value >= 0 else "🔴"
            st.write(f"- Performance: **{color} {selected_data['Performance']}**")
        else:
            st.write(f"- Performance: **{selected_data['Performance']}**")

    # Dividend sectie
    st.divider()
    st.write("### 💰 Dividend Beheer")

    # Formulier om dividend toe te voegen
    with st.expander("➕ Nieuw dividend toevoegen", expanded=False):
        with st.form(f"dividend_form_{selected_ticker}"):
            col1, col2 = st.columns(2)

            with col1:
                div_date = st.date_input("Ex-Dividend Datum", key=f"div_date_{selected_ticker}")
                div_amount = st.number_input(
                    "Bruto bedrag",
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                    help="Voer het totale bruto dividend in (voor al je aandelen)",
                    key=f"div_amount_{selected_ticker}"
                )
                div_currency = st.selectbox(
                    "Currency",
                    options=["EUR", "USD", "GBP", "CHF", "CAD", "AUD", "JPY"],
                    index=0,
                    key=f"div_currency_{selected_ticker}"
                )

            with col2:
                div_notes = st.text_area(
                    "Notities (optioneel)",
                    height=40,
                    key=f"div_notes_{selected_ticker}"
                )
                div_received = st.checkbox(
                    "✅ Ontvangen",
                    value=False,
                    help="Vink aan als je dit dividend al ontvangen hebt",
                    key=f"div_received_{selected_ticker}"
                )
                div_tax_paid = st.checkbox(
                    "💰 Tax betaald (30%)",
                    value=False,
                    help="Vink aan als je roerende voorheffing (30%) betaald hebt op dit dividend (⚠️ Kan alleen als dividend ontvangen is)",
                    key=f"div_tax_{selected_ticker}"
                )

            st.info("ℹ️ Tax kan alleen betaald zijn als dividend ontvangen is")

            div_submitted = st.form_submit_button("💾 Dividend opslaan")

            if div_submitted and div_amount > 0:
                # Logica: tax_paid kan alleen true zijn als received ook true is
                final_tax_paid = div_tax_paid and div_received

                if div_tax_paid and not div_received:
                    st.warning("⚠️ Tax betaald is uitgezet omdat dividend nog niet ontvangen is")

                add_dividend(
                    selected_ticker,
                    selected_data['ISIN'],
                    div_date,
                    div_amount,
                    div_notes,
                    div_currency,
                    final_tax_paid,
                    div_received
                )
                st.success(f"✓ Dividend van {format_currency(div_amount, div_currency)} toegevoegd!")
                st.rerun()

    # Haal bestaande dividenden op
    dividends_df = get_manual_dividends(selected_ticker)

    if not dividends_df.empty:
        st.write("#### Ontvangen Dividenden")

        # Bereken totalen
        div_info = calculate_total_dividends(selected_ticker)
        TAX_RATE = 0.30

        # Maak tabel met actie knoppen
        div_table_data = []
        for _, row in dividends_df.iterrows():
            bruto = row['bruto_amount']
            currency = row.get('currency', 'EUR')
            tax_paid = row.get('tax_paid', 1)
            received = row.get('received', 0)

            tax = (bruto * TAX_RATE) if tax_paid else 0
            netto = bruto - tax

            div_table_data.append({
                'ID': row['id'],
                'Ex-Dividend Datum': row['ex_date'],
                'Currency': currency,
                'Bruto': format_currency(bruto, currency),
                'Tax (30%)': format_currency(tax, currency) if tax_paid else '-',
                'Netto': format_currency(netto, currency),
                'Tax betaald': "✅" if tax_paid else "❌",
                'Ontvangen': "✅" if received else "❌",
                'Notities': row['notes'] if row['notes'] else '-'
            })

        div_display_df = pd.DataFrame(div_table_data)

        # Toon tabel (zonder ID kolom in display)
        st.dataframe(
            div_display_df[['Ex-Dividend Datum', 'Currency', 'Bruto', 'Tax (30%)', 'Netto', 'Tax betaald', 'Ontvangen', 'Notities']],
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

        # Toon totalen - ALLEEN ontvangen dividenden
        st.divider()
        st.write(f"### 📊 Ontvangen Dividenden ({div_info['received_count']}/{div_info['count']})")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Totaal Bruto (ontvangen)", f"€{div_info['total_bruto']:.2f}")

        with col2:
            st.metric("Roerende Voorheffing (30%)", f"€{div_info['total_tax']:.2f}", delta=None, delta_color="off")

        with col3:
            st.metric("Totaal Netto (ontvangen)", f"€{div_info['total_netto']:.2f}", delta=f"+{div_info['total_netto']:.2f}")

        # Toon impact op totale performance
        if '_perf' in selected_data and div_info['total_netto'] > 0:
            st.info(f"💡 Je totale winst/verlies (inclusief €{div_info['total_netto']:.2f} netto dividend) is: **€{selected_data['_perf']['total_gain_loss']:.2f}** ({selected_data['_perf']['gain_loss_percent']:+.2f}%)")
    else:
        st.info("Nog geen dividenden toegevoegd. Gebruik het formulier hierboven om dividenden toe te voegen.")

    # Haal gedetailleerde transactie geschiedenis op
    st.divider()
    st.write("### 📜 Transactie Geschiedenis")

    conn = get_connection()
    tx_query = f"""
    SELECT
        date as Datum,
        transaction_type as Type,
        quantity as Aantal,
        price_per_share as 'Prijs/stuk',
        fees as Kosten,
        (quantity * price_per_share + fees) as Totaal
    FROM transactions
    WHERE ticker = '{selected_ticker}'
    ORDER BY date DESC
    """

    tx_df = pd.read_sql_query(tx_query, conn)
    conn.close()

    st.dataframe(tx_df, use_container_width=True, hide_index=True)

    # Refresh button
    if st.button("🔄 Ververs prijzen", type="primary"):
        st.rerun()
