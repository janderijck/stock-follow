import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import calendar

DB_PATH = Path("data.db")

st.set_page_config(page_title="Dividend Kalender", page_icon="📅", layout="wide")


# --------- Database helpers ---------
def get_connection():
    """Maakt verbinding met de database."""
    return sqlite3.connect(DB_PATH)


def get_all_dividends():
    """Haalt alle dividend uitkeringen op."""
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT
            d.ticker,
            d.isin,
            d.ex_date,
            d.bruto_amount,
            d.notes,
            t.name
        FROM dividends d
        LEFT JOIN (
            SELECT DISTINCT ticker, name FROM transactions
        ) t ON d.ticker = t.ticker
        ORDER BY d.ex_date DESC
        """,
        conn
    )
    conn.close()

    if not df.empty:
        df['ex_date'] = pd.to_datetime(df['ex_date'])

    return df


def get_active_portfolio_tickers():
    """Haalt alle actieve tickers op (met positieve holdings)."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT ticker, name
        FROM transactions
        WHERE ticker IN (
            SELECT ticker
            FROM transactions
            GROUP BY ticker
            HAVING SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) > 0
        )
    """)

    result = cursor.fetchall()
    conn.close()

    return {row[0]: row[1] for row in result}


def predict_next_dividend(ticker):
    """Voorspelt volgende dividend datum op basis van historische patronen."""
    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT ex_date, bruto_amount
        FROM dividends
        WHERE ticker = ?
        ORDER BY ex_date DESC
        LIMIT 4
        """,
        conn,
        params=(ticker,)
    )
    conn.close()

    if len(df) < 2:
        return None

    df['ex_date'] = pd.to_datetime(df['ex_date'])

    # Bereken gemiddelde interval tussen dividenden
    intervals = []
    for i in range(len(df) - 1):
        delta = (df.iloc[i]['ex_date'] - df.iloc[i+1]['ex_date']).days
        intervals.append(delta)

    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        last_date = df.iloc[0]['ex_date']
        next_predicted = last_date + timedelta(days=avg_interval)

        return {
            'predicted_date': next_predicted,
            'avg_interval_days': round(avg_interval),
            'last_amount': df.iloc[0]['bruto_amount']
        }

    return None


# --------- UI ---------
st.title("📅 Dividend Kalender")
st.write("Overzicht van dividend uitkeringen en voorspellingen")

# Tabs voor verschillende views
tab1, tab2, tab3 = st.tabs(["📊 Overzicht", "📅 Kalender View", "🔮 Voorspellingen"])

# Tab 1: Overzicht
with tab1:
    st.subheader("Alle Dividend Uitkeringen")

    dividends_df = get_all_dividends()

    if dividends_df.empty:
        st.info("Nog geen dividend uitkeringen geregistreerd.")
    else:
        # Filter opties
        col1, col2 = st.columns(2)

        with col1:
            year_filter = st.selectbox(
                "Filter op jaar:",
                options=['Alle'] + sorted(dividends_df['ex_date'].dt.year.unique().tolist(), reverse=True)
            )

        with col2:
            ticker_filter = st.selectbox(
                "Filter op aandeel:",
                options=['Alle'] + sorted(dividends_df['ticker'].unique().tolist())
            )

        # Apply filters
        filtered_df = dividends_df.copy()
        if year_filter != 'Alle':
            filtered_df = filtered_df[filtered_df['ex_date'].dt.year == year_filter]
        if ticker_filter != 'Alle':
            filtered_df = filtered_df[filtered_df['ticker'] == ticker_filter]

        # Display tabel
        TAX_RATE = 0.30

        display_data = []
        for _, row in filtered_df.iterrows():
            tax = row['bruto_amount'] * TAX_RATE
            netto = row['bruto_amount'] - tax

            display_data.append({
                'Ex-Dividend Datum': row['ex_date'].strftime('%Y-%m-%d'),
                'Aandeel': row['name'] if pd.notna(row['name']) else row['ticker'],
                'Ticker': row['ticker'],
                'Bruto': f"€{row['bruto_amount']:.2f}",
                'Tax (30%)': f"€{tax:.2f}",
                'Netto': f"€{netto:.2f}",
                'Notities': row['notes'] if pd.notna(row['notes']) and row['notes'] else '-'
            })

        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Statistieken
        st.divider()
        col1, col2, col3 = st.columns(3)

        total_bruto = filtered_df['bruto_amount'].sum()
        total_tax = total_bruto * TAX_RATE
        total_netto = total_bruto - total_tax

        with col1:
            st.metric("Totaal Bruto", f"€{total_bruto:.2f}")

        with col2:
            st.metric("Totaal Tax (30%)", f"€{total_tax:.2f}")

        with col3:
            st.metric("Totaal Netto", f"€{total_netto:.2f}")

# Tab 2: Kalender View
with tab2:
    st.subheader("Kalender View")

    dividends_df = get_all_dividends()

    if dividends_df.empty:
        st.info("Nog geen dividend uitkeringen geregistreerd.")
    else:
        # Selecteer jaar en maand
        col1, col2 = st.columns(2)

        with col1:
            years = sorted(dividends_df['ex_date'].dt.year.unique().tolist(), reverse=True)
            current_year = datetime.now().year
            selected_year = st.selectbox(
                "Jaar:",
                options=years,
                index=years.index(current_year) if current_year in years else 0
            )

        with col2:
            selected_month = st.selectbox(
                "Maand:",
                options=range(1, 13),
                format_func=lambda x: calendar.month_name[x],
                index=datetime.now().month - 1
            )

        # Filter op geselecteerde maand
        month_filter = dividends_df[
            (dividends_df['ex_date'].dt.year == selected_year) &
            (dividends_df['ex_date'].dt.month == selected_month)
        ]

        if month_filter.empty:
            st.info(f"Geen dividenden in {calendar.month_name[selected_month]} {selected_year}")
        else:
            st.write(f"### {calendar.month_name[selected_month]} {selected_year}")

            # Groepeer per dag
            for day in sorted(month_filter['ex_date'].dt.day.unique()):
                day_data = month_filter[month_filter['ex_date'].dt.day == day]

                with st.expander(f"📅 {day} {calendar.month_name[selected_month]} - {len(day_data)} dividend(en)", expanded=True):
                    for _, row in day_data.iterrows():
                        tax = row['bruto_amount'] * 0.30
                        netto = row['bruto_amount'] - tax

                        st.write(f"**{row['name'] if pd.notna(row['name']) else row['ticker']}** ({row['ticker']})")
                        st.write(f"- Bruto: €{row['bruto_amount']:.2f} | Tax: €{tax:.2f} | Netto: €{netto:.2f}")
                        if pd.notna(row['notes']) and row['notes']:
                            st.write(f"- Notities: {row['notes']}")
                        st.divider()

# Tab 3: Voorspellingen
with tab3:
    st.subheader("🔮 Voorspelde Volgende Dividenden")
    st.info("Voorspellingen gebaseerd op historische patronen uit je dividend data.")

    active_tickers = get_active_portfolio_tickers()

    if not active_tickers:
        st.warning("Geen actieve posities in je portfolio.")
    else:
        predictions = []

        for ticker, name in active_tickers.items():
            prediction = predict_next_dividend(ticker)

            if prediction:
                predictions.append({
                    'Aandeel': name,
                    'Ticker': ticker,
                    'Voorspelde Datum': prediction['predicted_date'].strftime('%Y-%m-%d'),
                    'Gemiddeld Interval': f"{prediction['avg_interval_days']} dagen",
                    'Laatste Bedrag': f"€{prediction['last_amount']:.2f}",
                    'Dagen tot uitkering': (prediction['predicted_date'] - datetime.now()).days
                })

        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_df = pred_df.sort_values('Voorspelde Datum')

            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            st.divider()
            st.warning("⚠️ Let op: Dit zijn schattingen op basis van je historische data. Werkelijke ex-dividend datums kunnen afwijken. Controleer altijd de officiële bronnen!")

            # Upcoming dividends (binnen 90 dagen)
            upcoming = [p for p in predictions if 0 <= p['Dagen tot uitkering'] <= 90]

            if upcoming:
                st.write("### 📌 Aankomende Dividenden (binnen 90 dagen)")
                for p in sorted(upcoming, key=lambda x: x['Dagen tot uitkering']):
                    st.success(f"**{p['Aandeel']}** ({p['Ticker']}) - {p['Voorspelde Datum']} (over {p['Dagen tot uitkering']} dagen)")
        else:
            st.info("Niet genoeg historische data om voorspellingen te maken. Voeg minstens 2 dividenden per aandeel toe.")

# Nuttige links en bronnen
st.divider()
with st.expander("📚 Bronnen voor Dividend Informatie"):
    st.write("""
    **Waar vind je ex-dividend datums?**

    - **Yahoo Finance**: ticker opzoeken → "Historical Data" → "Dividends"
    - **Investor Relations**: Officiële website van het bedrijf
    - **Degiro/Interactive Brokers**: In je broker platform
    - **DividendMax.com**: Gratis dividend kalender
    - **Investing.com**: Dividend kalender sectie

    **Tips:**
    - Ex-dividend datum = laatste dag om recht te hebben op dividend
    - Payment datum = wanneer dividend wordt uitbetaald (meestal 2-4 weken later)
    - Voeg dividenden toe zodra je ze ontvangt voor accurate tracking
    """)
