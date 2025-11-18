import streamlit as st
import sqlite3
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data.db")

st.set_page_config(page_title="Portfolio Overzicht", page_icon="📊", layout="wide")


# --------- Database helpers ---------
def get_connection():
    """Maakt verbinding met de database."""
    return sqlite3.connect(DB_PATH)


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


def get_current_price(ticker):
    """
    Haalt de huidige prijs op voor een ticker via yfinance API.
    Retourneert dict met current_price, change_percent, currency.
    """
    if not ticker:
        return None

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Probeer verschillende velden voor de huidige prijs
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')

        if current_price is None:
            # Probeer via history als info niet werkt
            hist = stock.history(period='1d')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]

        if current_price is None:
            return {"error": "Geen prijs beschikbaar"}

        # Haal percentage verandering op (vandaag)
        change_percent = info.get('regularMarketChangePercent', 0)
        currency = info.get('currency', 'USD')

        return {
            'current_price': float(current_price),
            'change_percent': float(change_percent) if change_percent else 0,
            'currency': currency,
            'name': info.get('longName', info.get('shortName', ticker))
        }

    except Exception as e:
        return {"error": f"Fout bij ophalen prijs: {str(e)}"}


def calculate_performance(avg_purchase_price, current_price, quantity):
    """Berekent winst/verlies statistieken."""
    total_invested = avg_purchase_price * quantity
    current_value = current_price * quantity
    total_gain_loss = current_value - total_invested
    gain_loss_percent = ((current_price - avg_purchase_price) / avg_purchase_price) * 100

    return {
        'total_invested': total_invested,
        'current_value': current_value,
        'total_gain_loss': total_gain_loss,
        'gain_loss_percent': gain_loss_percent
    }


# --------- UI ---------
st.title("📊 Portfolio Overzicht")

# Haal portfolio data op
holdings = get_portfolio_holdings()

if holdings.empty:
    st.info("Je portfolio is nog leeg. Voeg eerst transacties toe op de hoofdpagina.")
    st.stop()

# Initialize session state voor geselecteerd aandeel
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

# Toon portfolio overzicht
st.subheader("Jouw Holdings")

# Bereken totalen
portfolio_data = []

with st.spinner("Huidige prijzen ophalen..."):
    for idx, row in holdings.iterrows():
        ticker = row['ticker']

        # Haal huidige prijs op
        price_info = get_current_price(ticker)

        if price_info and 'error' not in price_info:
            current_price = price_info['current_price']

            # Bereken performance
            perf = calculate_performance(
                row['avg_purchase_price'],
                current_price,
                row['total_quantity']
            )

            portfolio_data.append({
                'Aandeel': row['name'],
                'Ticker': ticker,
                'ISIN': row['isin'],
                'Aantal': int(row['total_quantity']),
                'Avg Aankoopprijs': f"€{row['avg_purchase_price']:.2f}",
                'Huidige Prijs': f"€{current_price:.2f}",
                'Totaal Geinvesteerd': f"€{perf['total_invested']:.2f}",
                'Huidige Waarde': f"€{perf['current_value']:.2f}",
                'Winst/Verlies': f"€{perf['total_gain_loss']:.2f}",
                'Performance': f"{perf['gain_loss_percent']:+.2f}%",
                '_current_price': current_price,
                '_perf': perf,
                '_avg_price': row['avg_purchase_price'],
                '_quantity': row['total_quantity']
            })
        else:
            # Kon prijs niet ophalen
            portfolio_data.append({
                'Aandeel': row['name'],
                'Ticker': ticker,
                'ISIN': row['isin'],
                'Aantal': int(row['total_quantity']),
                'Avg Aankoopprijs': f"€{row['avg_purchase_price']:.2f}",
                'Huidige Prijs': "N/A",
                'Totaal Geinvesteerd': f"€{row['total_invested_with_fees']:.2f}",
                'Huidige Waarde': "N/A",
                'Winst/Verlies': "N/A",
                'Performance': "N/A",
                '_error': price_info.get('error', 'Onbekende fout') if price_info else 'API error'
            })

# Maak DataFrame voor display
portfolio_df = pd.DataFrame(portfolio_data)

# Toon portfolio tabel (zonder hidden columns)
display_columns = ['Aandeel', 'Ticker', 'Aantal', 'Avg Aankoopprijs', 'Huidige Prijs',
                   'Totaal Geinvesteerd', 'Huidige Waarde', 'Winst/Verlies', 'Performance']

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

    # Haal gedetailleerde transactie geschiedenis op
    st.write("### Transactie Geschiedenis")

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
