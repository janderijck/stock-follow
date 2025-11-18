import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import calendar
import requests

DB_PATH = Path("data.db")

st.set_page_config(page_title="Dividend Kalender", page_icon="📅", layout="wide")


# --------- Database helpers ---------
def get_connection():
    """Maakt verbinding met de database."""
    conn = sqlite3.connect(DB_PATH)
    # Zorg dat settings tabel bestaat
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    return conn


def get_setting(key, default=None):
    """Haalt een setting op uit de database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else default


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


def fetch_dividends_from_alphavantage(ticker):
    """Haalt dividend geschiedenis op van Alpha Vantage API."""
    api_key = get_setting('alpha_vantage_api_key', '')

    if not api_key:
        return {'error': 'Geen Alpha Vantage API key ingesteld. Ga naar Admin > API Settings.'}

    try:
        # Use TIME_SERIES_DAILY_ADJUSTED which includes dividend amounts
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}"
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            return {'error': f'API error: status {response.status_code}'}

        data = response.json()

        # Check for API errors
        if 'Error Message' in data:
            return {'error': data['Error Message']}

        if 'Note' in data:
            return {'error': 'API rate limit bereikt. Probeer het later opnieuw.'}

        if 'Time Series (Daily)' not in data:
            return {'error': 'Geen data beschikbaar voor dit aandeel'}

        # Parse dividend data from time series
        time_series = data['Time Series (Daily)']
        dividends = []

        for date_str, daily_data in time_series.items():
            dividend_amount = float(daily_data.get('7. dividend amount', 0))

            # Only add dates where dividend was paid (amount > 0)
            if dividend_amount > 0:
                dividends.append({
                    'ex_date': date_str,
                    'amount': dividend_amount
                })

        # Sort by date descending (most recent first)
        dividends.sort(key=lambda x: x['ex_date'], reverse=True)

        return {'dividends': dividends}

    except requests.exceptions.Timeout:
        return {'error': 'API timeout - probeer het opnieuw'}
    except requests.exceptions.RequestException as e:
        return {'error': f'Netwerk error: {str(e)}'}
    except Exception as e:
        return {'error': f'Onverwachte fout: {str(e)}'}


def import_dividend_to_db(ticker, isin, ex_date, amount):
    """Importeert een dividend uitkering in de database."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check of deze dividend al bestaat
    cursor.execute(
        "SELECT id FROM dividends WHERE ticker = ? AND ex_date = ?",
        (ticker, ex_date)
    )

    if cursor.fetchone():
        conn.close()
        return False  # Al bestaand

    # Insert nieuwe dividend
    cursor.execute(
        """
        INSERT INTO dividends (ticker, isin, ex_date, bruto_amount, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ticker, isin, ex_date, amount, "Automatisch geïmporteerd van Alpha Vantage")
    )

    conn.commit()
    conn.close()
    return True  # Nieuw toegevoegd


def get_portfolio_stocks_with_isin():
    """Haalt alle unieke aandelen op uit de portfolio met ISIN."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT ticker, name, isin
        FROM transactions
        ORDER BY ticker
    """)

    result = cursor.fetchall()
    conn.close()

    return [{'ticker': row[0], 'name': row[1], 'isin': row[2]} for row in result]


# --------- UI ---------
st.title("📅 Dividend Kalender")
st.write("Overzicht van dividend uitkeringen en voorspellingen")

# Tabs voor verschillende views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overzicht", "📅 Kalender View", "🔮 Voorspellingen", "⬇️ Import Dividenden"])

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

# Tab 4: Import Dividenden
with tab4:
    st.subheader("⬇️ Dividend Geschiedenis Importeren")
    st.info("📡 Importeer automatisch dividend geschiedenis van Alpha Vantage voor je portfolio aandelen.")

    # Check of API key is ingesteld
    api_key = get_setting('alpha_vantage_api_key', '')

    if not api_key:
        st.warning("⚠️ Geen Alpha Vantage API key ingesteld!")
        st.write("Ga naar **Admin > API Settings** om je API key in te stellen.")
        st.write("[Vraag gratis API key aan](https://www.alphavantage.co/support/#api-key)")
    else:
        st.success("✓ Alpha Vantage API key actief")

        # Haal portfolio aandelen op
        portfolio_stocks = get_portfolio_stocks_with_isin()

        if not portfolio_stocks:
            st.warning("Geen aandelen gevonden in je portfolio.")
        else:
            st.write(f"**{len(portfolio_stocks)} aandelen gevonden in je portfolio**")

            # Selecteer aandeel
            selected_stock = st.selectbox(
                "Selecteer aandeel om dividenden te importeren:",
                options=portfolio_stocks,
                format_func=lambda x: f"{x['name']} ({x['ticker']})"
            )

            if selected_stock:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Geselecteerd:** {selected_stock['name']} ({selected_stock['ticker']})")

                with col2:
                    if st.button("📥 Importeer Dividenden", type="primary"):
                        with st.spinner(f"Dividenden ophalen voor {selected_stock['ticker']}..."):
                            result = fetch_dividends_from_alphavantage(selected_stock['ticker'])

                            if 'error' in result:
                                st.error(f"❌ {result['error']}")
                            elif 'dividends' in result:
                                dividends = result['dividends']

                                if not dividends:
                                    st.info(f"Geen dividend geschiedenis gevonden voor {selected_stock['ticker']}")
                                else:
                                    st.success(f"✓ {len(dividends)} dividenden gevonden!")

                                    # Importeer dividenden
                                    imported_count = 0
                                    skipped_count = 0

                                    with st.spinner("Dividenden importeren..."):
                                        for div in dividends:
                                            was_imported = import_dividend_to_db(
                                                selected_stock['ticker'],
                                                selected_stock['isin'],
                                                div['ex_date'],
                                                div['amount']
                                            )

                                            if was_imported:
                                                imported_count += 1
                                            else:
                                                skipped_count += 1

                                    st.success(f"✓ Import voltooid! {imported_count} nieuwe dividenden toegevoegd.")

                                    if skipped_count > 0:
                                        st.info(f"ℹ️ {skipped_count} dividenden overgeslagen (al in database)")

                                    st.rerun()

            st.divider()

            # Bulk import optie
            st.write("### 🚀 Bulk Import")
            st.write("Importeer dividenden voor alle aandelen in je portfolio in één keer.")

            if st.button("📥 Importeer Alle Dividenden", type="secondary"):
                total_imported = 0
                total_skipped = 0
                errors = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, stock in enumerate(portfolio_stocks):
                    status_text.write(f"Bezig met {stock['name']} ({stock['ticker']})...")
                    progress_bar.progress((idx + 1) / len(portfolio_stocks))

                    result = fetch_dividends_from_alphavantage(stock['ticker'])

                    if 'error' in result:
                        errors.append(f"{stock['ticker']}: {result['error']}")
                    elif 'dividends' in result:
                        for div in result['dividends']:
                            was_imported = import_dividend_to_db(
                                stock['ticker'],
                                stock['isin'],
                                div['ex_date'],
                                div['amount']
                            )

                            if was_imported:
                                total_imported += 1
                            else:
                                total_skipped += 1

                status_text.empty()
                progress_bar.empty()

                st.success(f"✓ Bulk import voltooid! {total_imported} nieuwe dividenden toegevoegd.")

                if total_skipped > 0:
                    st.info(f"ℹ️ {total_skipped} dividenden overgeslagen (al in database)")

                if errors:
                    with st.expander("⚠️ Errors tijdens import", expanded=False):
                        for error in errors:
                            st.warning(error)

                st.rerun()

            st.divider()
            st.warning("""
            **Let op:**
            - Alpha Vantage heeft rate limits (25 API calls per dag voor gratis accounts)
            - Niet alle aandelen zijn beschikbaar in Alpha Vantage (vooral Europese aandelen kunnen ontbreken)
            - De bulk import kan enkele minuten duren
            """)

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
