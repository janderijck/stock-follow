import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data.db")

st.set_page_config(page_title="Gesloten Posities", page_icon="📕", layout="wide")


# --------- Database helpers ---------
def get_connection():
    """Maakt verbinding met de database."""
    return sqlite3.connect(DB_PATH)


def get_closed_positions():
    """
    Haalt alle gesloten posities op (waar alle aandelen verkocht zijn).
    Berekent gerealiseerde winst/verlies.
    """
    conn = get_connection()

    # Query voor gesloten posities
    query = """
    WITH position_summary AS (
        SELECT
            ticker,
            isin,
            name,
            SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE 0 END) as total_bought,
            SUM(CASE WHEN transaction_type = 'SELL' THEN quantity ELSE 0 END) as total_sold,
            SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price_per_share + fees ELSE 0 END) as total_invested,
            SUM(CASE WHEN transaction_type = 'SELL' THEN quantity * price_per_share - fees ELSE 0 END) as total_proceeds,
            MIN(CASE WHEN transaction_type = 'BUY' THEN date END) as first_buy_date,
            MAX(CASE WHEN transaction_type = 'SELL' THEN date END) as last_sell_date,
            currency
        FROM transactions
        GROUP BY ticker, isin, name, currency
        HAVING total_bought = total_sold AND total_bought > 0
    )
    SELECT * FROM position_summary
    ORDER BY last_sell_date DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if not df.empty:
        # Bereken gerealiseerde winst/verlies
        df['realized_gain_loss'] = df['total_proceeds'] - df['total_invested']
        df['realized_gain_loss_percent'] = (df['realized_gain_loss'] / df['total_invested']) * 100

    return df


def get_dividends_for_ticker(ticker):
    """Haalt totaal dividend op voor een ticker."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COALESCE(SUM(bruto_amount), 0) as total_bruto
        FROM dividends
        WHERE ticker = ?
        """,
        (ticker,)
    )

    result = cursor.fetchone()
    conn.close()

    total_bruto = result[0] if result[0] else 0
    TAX_RATE = 0.30
    total_netto = total_bruto * (1 - TAX_RATE)

    return {
        'total_bruto': total_bruto,
        'total_netto': total_netto
    }


def get_transaction_history(ticker):
    """Haalt alle transacties op voor een ticker."""
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT
            date as Datum,
            transaction_type as Type,
            quantity as Aantal,
            price_per_share as 'Prijs/stuk',
            fees as Kosten,
            broker as Broker
        FROM transactions
        WHERE ticker = ?
        ORDER BY date
        """,
        conn,
        params=(ticker,)
    )
    conn.close()
    return df


# --------- UI ---------
st.title("📕 Gesloten Posities")
st.write("Overzicht van volledig verkochte posities met gerealiseerde winsten/verliezen")

# Haal gesloten posities op
closed_positions = get_closed_positions()

if closed_positions.empty:
    st.info("Je hebt nog geen volledig verkochte posities.")
    st.stop()

# Bereken totalen
total_invested_all = closed_positions['total_invested'].sum()
total_proceeds_all = closed_positions['total_proceeds'].sum()
total_realized_gain_loss = closed_positions['realized_gain_loss'].sum()

# Toon algemene metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Totaal Geïnvesteerd", f"€{total_invested_all:,.2f}")

with col2:
    st.metric("Totaal Opbrengst", f"€{total_proceeds_all:,.2f}")

with col3:
    gain_loss_color = "normal" if total_realized_gain_loss >= 0 else "inverse"
    st.metric(
        "Gerealiseerde Winst/Verlies",
        f"€{total_realized_gain_loss:,.2f}",
        delta=f"{(total_realized_gain_loss / total_invested_all * 100):+.2f}%" if total_invested_all > 0 else "0%"
    )

with col4:
    st.metric("Aantal Gesloten Posities", len(closed_positions))

st.divider()

# Toon gesloten posities tabel
st.subheader("Gesloten Posities")

# Maak display tabel
display_data = []
for _, row in closed_positions.iterrows():
    # Haal dividend info op
    div_info = get_dividends_for_ticker(row['ticker'])

    # Bereken totale return inclusief dividenden
    total_return_incl_div = row['realized_gain_loss'] + div_info['total_netto']
    total_return_percent_incl_div = (total_return_incl_div / row['total_invested']) * 100 if row['total_invested'] > 0 else 0

    display_data.append({
        'Aandeel': row['name'],
        'Ticker': row['ticker'],
        'Aandelen': int(row['total_bought']),
        'Gekocht': f"€{row['total_invested']:.2f}",
        'Verkocht': f"€{row['total_proceeds']:.2f}",
        'Winst/Verlies': f"€{row['realized_gain_loss']:.2f}",
        'Return %': f"{row['realized_gain_loss_percent']:+.2f}%",
        'Dividend': f"€{div_info['total_netto']:.2f}",
        'Totaal Return': f"€{total_return_incl_div:.2f}",
        'Totaal %': f"{total_return_percent_incl_div:+.2f}%",
        'Periode': f"{row['first_buy_date']} tot {row['last_sell_date']}",
        '_ticker': row['ticker']
    })

display_df = pd.DataFrame(display_data)

st.dataframe(
    display_df[[col for col in display_df.columns if not col.startswith('_')]],
    use_container_width=True,
    hide_index=True
)

# Selecteer positie voor details
st.divider()
st.subheader("Details Gesloten Positie")

selected = st.selectbox(
    "Selecteer een gesloten positie:",
    options=display_df['_ticker'].tolist(),
    format_func=lambda x: f"{display_df[display_df['_ticker']==x]['Aandeel'].values[0]} ({x})"
)

if selected:
    selected_row = display_df[display_df['_ticker'] == selected].iloc[0]

    st.write(f"### {selected_row['Aandeel']} ({selected})")

    # Metrics voor geselecteerde positie
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Totaal Geïnvesteerd", selected_row['Gekocht'])

    with col2:
        st.metric("Totaal Verkocht", selected_row['Verkocht'])

    with col3:
        st.metric("Winst/Verlies", selected_row['Winst/Verlies'], delta=selected_row['Return %'])

    # Toon transactie geschiedenis
    st.write("#### Transactie Geschiedenis")
    tx_history = get_transaction_history(selected)
    st.dataframe(tx_history, use_container_width=True, hide_index=True)

    # Toon dividend informatie indien van toepassing
    div_info = get_dividends_for_ticker(selected)
    if div_info['total_netto'] > 0:
        st.write("#### Dividend Informatie")
        st.info(f"💰 Totaal ontvangen dividend (netto): **€{div_info['total_netto']:.2f}**")

    # Totale performance
    st.divider()
    st.success(f"🎯 **Totale return (incl. dividend): {selected_row['Totaal Return']} ({selected_row['Totaal %']})**")
