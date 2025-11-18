import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import calendar
import requests
import yfinance as yf

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
    # Zorg dat dividend cache tabel bestaat
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dividend_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            isin TEXT,
            ex_date TEXT NOT NULL,
            amount REAL NOT NULL,
            currency TEXT DEFAULT 'USD',
            fetched_at TEXT NOT NULL,
            UNIQUE(ticker, ex_date)
        )
    """)

    # Migratie: Voeg 'received' kolom toe aan dividends tabel als deze nog niet bestaat
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(dividends)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'received' not in columns:
        cursor.execute("ALTER TABLE dividends ADD COLUMN received INTEGER DEFAULT 0")
        conn.commit()

    if 'currency' not in columns:
        cursor.execute("ALTER TABLE dividends ADD COLUMN currency TEXT DEFAULT 'USD'")
        conn.commit()

    # Migratie: Voeg 'currency' kolom toe aan dividend_cache als deze nog niet bestaat
    cursor.execute("PRAGMA table_info(dividend_cache)")
    cache_columns = [col[1] for col in cursor.fetchall()]

    if 'currency' not in cache_columns:
        cursor.execute("ALTER TABLE dividend_cache ADD COLUMN currency TEXT DEFAULT 'USD'")
        conn.commit()

    return conn


def get_setting(key, default=None):
    """Haalt een setting op uit de database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else default


def format_currency(amount, currency='USD'):
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

    # Voor symbolen die achter het bedrag staan (zoals EUR)
    if currency == 'EUR':
        return f"€{amount:.2f}"
    else:
        # Voor symbolen die voor het bedrag staan (zoals USD)
        return f"{symbol}{amount:.2f}"


def update_dividend_received_status(dividend_id, received):
    """Update de received status van een dividend."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE dividends SET received = ? WHERE id = ?",
        (1 if received else 0, dividend_id)
    )
    conn.commit()
    conn.close()


def get_all_dividends():
    """Haalt alle dividend uitkeringen op."""
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT
            d.id,
            d.ticker,
            d.isin,
            d.ex_date,
            d.bruto_amount,
            d.currency,
            d.notes,
            d.received,
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


def get_currency_for_ticker(ticker):
    """Detecteert de currency voor een ticker uit transactions tabel of yfinance."""
    conn = get_connection()
    cursor = conn.cursor()

    # Probeer eerst uit transactions tabel
    cursor.execute(
        "SELECT currency FROM transactions WHERE ticker = ? LIMIT 1",
        (ticker,)
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]

    # Als niet in transactions, probeer uit yfinance
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        currency = info.get('currency', 'USD')
        return currency
    except:
        # Default naar USD als alles faalt
        return 'USD'


def fetch_dividends_from_yfinance(ticker, debug=False):
    """Haalt dividend geschiedenis op van Yahoo Finance via yfinance."""
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)

        # Get dividend history
        dividends_series = stock.dividends

        # Get currency
        currency = get_currency_for_ticker(ticker)

        if debug:
            return {
                'debug': {
                    'ticker': ticker,
                    'currency': currency,
                    'dividends_count': len(dividends_series),
                    'dividends_series': dividends_series.to_dict(),
                    'info_keys': list(stock.info.keys()) if hasattr(stock, 'info') else []
                }
            }

        if dividends_series.empty:
            return {'error': f'Geen dividend geschiedenis gevonden voor {ticker}'}

        # Convert to list of dicts
        dividends = []
        for date, amount in dividends_series.items():
            dividends.append({
                'ex_date': date.strftime('%Y-%m-%d'),
                'amount': float(amount)
            })

        # Sort by date descending (most recent first)
        dividends.sort(key=lambda x: x['ex_date'], reverse=True)

        return {'dividends': dividends, 'currency': currency}

    except Exception as e:
        return {'error': f'Fout bij ophalen van dividend data: {str(e)}'}


def fetch_dividends_from_alphavantage(ticker, debug=False):
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

        # Debug mode: return raw response
        if debug:
            return {'debug': data, 'url': url}

        # Check for API errors
        if 'Error Message' in data:
            return {'error': data['Error Message']}

        if 'Note' in data:
            return {'error': 'API rate limit bereikt. Probeer het later opnieuw.'}

        if 'Information' in data:
            return {'error': f"API info: {data['Information']}"}

        if 'Time Series (Daily)' not in data:
            # Show what keys we got instead
            available_keys = ', '.join(data.keys())
            return {'error': f'Geen "Time Series (Daily)" in response. Beschikbare keys: {available_keys}'}

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


def save_dividends_to_cache(ticker, isin, dividends, currency='USD'):
    """Slaat dividend data op in de cache buffer tabel."""
    conn = get_connection()
    cursor = conn.cursor()

    fetched_at = datetime.now().isoformat()
    added_count = 0
    updated_count = 0

    for div in dividends:
        try:
            cursor.execute(
                """
                INSERT INTO dividend_cache (ticker, isin, ex_date, amount, currency, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, ex_date)
                DO UPDATE SET amount = ?, currency = ?, fetched_at = ?
                """,
                (ticker, isin, div['ex_date'], div['amount'], currency, fetched_at, div['amount'], currency, fetched_at)
            )
            if cursor.rowcount == 1:
                added_count += 1
            else:
                updated_count += 1
        except Exception as e:
            st.error(f"Error opslaan dividend: {e}")

    conn.commit()
    conn.close()

    return {'added': added_count, 'updated': updated_count}


def get_cached_dividends(ticker=None):
    """Haalt cached dividend data op voor een specifieke ticker of alle tickers."""
    conn = get_connection()

    if ticker:
        query = """
            SELECT ticker, isin, ex_date, amount, currency, fetched_at
            FROM dividend_cache
            WHERE ticker = ?
            ORDER BY ex_date DESC
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
    else:
        query = """
            SELECT ticker, isin, ex_date, amount, currency, fetched_at
            FROM dividend_cache
            ORDER BY ticker, ex_date DESC
        """
        df = pd.read_sql_query(query, conn)

    conn.close()
    return df


def get_cache_last_updated(ticker):
    """Haalt de laatste update timestamp op voor een ticker."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT MAX(fetched_at) FROM dividend_cache WHERE ticker = ?",
        (ticker,)
    )
    result = cursor.fetchone()
    conn.close()

    return result[0] if result and result[0] else None


def clear_cache(ticker=None):
    """Leegt de cache voor een specifieke ticker of alle tickers."""
    conn = get_connection()
    cursor = conn.cursor()

    if ticker:
        cursor.execute("DELETE FROM dividend_cache WHERE ticker = ?", (ticker,))
    else:
        cursor.execute("DELETE FROM dividend_cache")

    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    return deleted


def import_from_cache_to_db(ticker, ex_date):
    """Importeert een specifieke dividend van cache naar de database."""
    conn = get_connection()
    cursor = conn.cursor()

    # Haal dividend op uit cache
    cursor.execute(
        """
        SELECT ticker, isin, ex_date, amount, currency
        FROM dividend_cache
        WHERE ticker = ? AND ex_date = ?
        """,
        (ticker, ex_date)
    )

    cache_row = cursor.fetchone()

    if not cache_row:
        conn.close()
        return {'success': False, 'error': 'Dividend niet gevonden in cache'}

    ticker, isin, ex_date, amount, currency = cache_row

    # Check of deze dividend al bestaat in de database
    cursor.execute(
        "SELECT id FROM dividends WHERE ticker = ? AND ex_date = ?",
        (ticker, ex_date)
    )

    if cursor.fetchone():
        conn.close()
        return {'success': False, 'error': 'Dividend bestaat al in database'}

    # Insert nieuwe dividend
    cursor.execute(
        """
        INSERT INTO dividends (ticker, isin, ex_date, bruto_amount, currency, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ticker, isin, ex_date, amount, currency, "Geïmporteerd van Yahoo Finance via cache")
    )

    conn.commit()
    conn.close()

    return {'success': True}


def import_all_from_cache_to_db(ticker):
    """Importeert alle dividenden van een ticker van cache naar database."""
    conn = get_connection()
    cursor = conn.cursor()

    # Haal alle cached dividenden op voor deze ticker
    cursor.execute(
        """
        SELECT ticker, isin, ex_date, amount, currency
        FROM dividend_cache
        WHERE ticker = ?
        """,
        (ticker,)
    )

    cache_rows = cursor.fetchall()

    if not cache_rows:
        conn.close()
        return {'imported': 0, 'skipped': 0}

    imported_count = 0
    skipped_count = 0

    for row in cache_rows:
        ticker, isin, ex_date, amount, currency = row

        # Check of deze dividend al bestaat
        cursor.execute(
            "SELECT id FROM dividends WHERE ticker = ? AND ex_date = ?",
            (ticker, ex_date)
        )

        if cursor.fetchone():
            skipped_count += 1
            continue

        # Insert nieuwe dividend
        cursor.execute(
            """
            INSERT INTO dividends (ticker, isin, ex_date, bruto_amount, currency, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker, isin, ex_date, amount, currency, "Geïmporteerd van Yahoo Finance via cache")
        )
        imported_count += 1

    conn.commit()
    conn.close()

    return {'imported': imported_count, 'skipped': skipped_count}


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
        (ticker, isin, ex_date, amount, "Automatisch geïmporteerd van Yahoo Finance")
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
            currency = row.get('currency', 'USD')

            # Check if past or future
            ex_date = pd.to_datetime(row['ex_date'])
            is_future = ex_date > pd.Timestamp.now()
            received_status = "🔮 Toekomst" if is_future else ("✅ Ja" if row.get('received', 0) else "❌ Nee")

            display_data.append({
                'Ex-Dividend Datum': row['ex_date'].strftime('%Y-%m-%d'),
                'Aandeel': row['name'] if pd.notna(row['name']) else row['ticker'],
                'Ticker': row['ticker'],
                'Currency': currency,
                'Bruto': format_currency(row['bruto_amount'], currency),
                'Tax (30%)': format_currency(tax, currency),
                'Netto': format_currency(netto, currency),
                'Ontvangen': received_status,
                'Notities': row['notes'] if pd.notna(row['notes']) and row['notes'] else '-'
            })

        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Statistieken
        st.divider()
        col1, col2, col3, col4 = st.columns(4)

        total_bruto = filtered_df['bruto_amount'].sum()
        total_tax = total_bruto * TAX_RATE
        total_netto = total_bruto - total_tax

        # Calculate received vs not received
        received_df = filtered_df[filtered_df['received'] == 1]
        received_netto = (received_df['bruto_amount'].sum() * (1 - TAX_RATE)) if not received_df.empty else 0

        with col1:
            st.metric("Totaal Bruto", f"€{total_bruto:.2f}")

        with col2:
            st.metric("Totaal Tax (30%)", f"€{total_tax:.2f}")

        with col3:
            st.metric("Totaal Netto", f"€{total_netto:.2f}")

        with col4:
            st.metric("Ontvangen (netto)", f"€{received_netto:.2f}", delta=f"{len(received_df)}/{len(filtered_df)}")

# Tab 2: Kalender View
with tab2:
    st.subheader("Kalender View")

    dividends_df = get_all_dividends()

    if dividends_df.empty:
        st.info("Nog geen dividend uitkeringen geregistreerd.")
    else:
        # Selecteer jaar en maand
        col1, col2, col3 = st.columns([2, 2, 1])

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

        st.divider()

        if month_filter.empty:
            st.info(f"Geen dividenden in {calendar.month_name[selected_month]} {selected_year}")
        else:
            # Maak visuele kalender grid
            st.write(f"### 📅 {calendar.month_name[selected_month]} {selected_year}")

            # Get calendar for selected month
            cal = calendar.monthcalendar(selected_year, selected_month)

            # Create dividend lookup
            dividend_days = {}
            for _, row in month_filter.iterrows():
                day = row['ex_date'].day
                if day not in dividend_days:
                    dividend_days[day] = []
                dividend_days[day].append(row)

            # Display calendar grid
            st.write("")

            # Legend
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption("✅ = Alle ontvangen")
            with col2:
                st.caption("⚠️ = Deels ontvangen")
            with col3:
                st.caption("❌ = Niet ontvangen")
            with col4:
                st.caption("🔮 = Toekomstig")

            st.write("")

            # Header with day names
            day_names = ['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo']
            cols = st.columns(7)
            for i, day_name in enumerate(day_names):
                with cols[i]:
                    st.markdown(f"**{day_name}**")

            # Calendar days
            for week in cal:
                cols = st.columns(7)
                for i, day in enumerate(week):
                    with cols[i]:
                        if day == 0:
                            st.write("")  # Empty day
                        else:
                            # Check if this day has dividends
                            if day in dividend_days:
                                # Day with dividend(s)
                                num_dividends = len(dividend_days[day])
                                total_amount = sum(d['bruto_amount'] for d in dividend_days[day])

                                # Check if past or future
                                date_check = datetime(selected_year, selected_month, day)
                                is_past = date_check < datetime.now()
                                is_future = date_check > datetime.now()

                                # Check if received (for past dividends)
                                all_received = all(d.get('received', 0) for d in dividend_days[day])
                                some_received = any(d.get('received', 0) for d in dividend_days[day])

                                # Determine button type and icon
                                if is_future:
                                    button_type = "secondary"
                                    icon = "🔮"
                                elif all_received:
                                    button_type = "primary"
                                    icon = "✅"
                                elif some_received:
                                    button_type = "primary"
                                    icon = "⚠️"
                                else:
                                    button_type = "primary"
                                    icon = "❌"

                                if st.button(
                                    f"**{day}**\n{icon} {num_dividends}x\n€{total_amount:.0f}",
                                    key=f"day_{selected_year}_{selected_month}_{day}",
                                    use_container_width=True,
                                    type=button_type
                                ):
                                    # Store selected day in session state
                                    st.session_state['selected_dividend_day'] = day
                            else:
                                # Regular day without dividend
                                st.markdown(f"<div style='text-align: center; padding: 10px; color: #666;'>{day}</div>", unsafe_allow_html=True)

            st.divider()

            # Show details for selected day
            selected_day = st.session_state.get('selected_dividend_day', None)

            if selected_day and selected_day in dividend_days:
                day_data = dividend_days[selected_day]

                # Check if this is a past or future date
                selected_date = datetime(selected_year, selected_month, selected_day)
                is_past = selected_date < datetime.now()
                is_future = selected_date > datetime.now()

                if is_past:
                    st.write(f"### 📋 Details voor {selected_day} {calendar.month_name[selected_month]} {selected_year} (Verleden)")
                elif is_future:
                    st.write(f"### 📋 Details voor {selected_day} {calendar.month_name[selected_month]} {selected_year} (Toekomst 🔮)")
                else:
                    st.write(f"### 📋 Details voor {selected_day} {calendar.month_name[selected_month]} {selected_year} (Vandaag)")

                for dividend in day_data:
                    tax = dividend['bruto_amount'] * 0.30
                    netto = dividend['bruto_amount'] - tax

                    with st.container():
                        col_name, col_received = st.columns([3, 1])

                        with col_name:
                            st.write(f"**{dividend['name'] if pd.notna(dividend['name']) else dividend['ticker']}** ({dividend['ticker']})")

                        with col_received:
                            # Only show checkbox for past/today dividends
                            if not is_future:
                                current_received = bool(dividend.get('received', 0))
                                received_checkbox = st.checkbox(
                                    "✅ Ontvangen",
                                    value=current_received,
                                    key=f"received_{dividend['id']}_{selected_year}_{selected_month}_{selected_day}"
                                )

                                # Update if changed
                                if received_checkbox != current_received:
                                    update_dividend_received_status(dividend['id'], received_checkbox)
                                    st.success("✓ Status bijgewerkt")
                                    st.rerun()
                            else:
                                st.info("🔮 Toekomstig")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Bruto", f"€{dividend['bruto_amount']:.2f}")
                        with col2:
                            st.metric("Tax (30%)", f"€{tax:.2f}")
                        with col3:
                            st.metric("Netto", f"€{netto:.2f}")

                        if pd.notna(dividend['notes']) and dividend['notes']:
                            st.info(f"📝 {dividend['notes']}")

                        st.divider()
            else:
                st.info("👆 Klik op een dag met dividend (💰) om details te zien")

            st.divider()

            # Also show list view below
            st.write("### 📋 Lijst View")

            # Groepeer per dag
            for day in sorted(month_filter['ex_date'].dt.day.unique()):
                day_data = month_filter[month_filter['ex_date'].dt.day == day]

                with st.expander(f"📅 {day} {calendar.month_name[selected_month]} - {len(day_data)} dividend(en)", expanded=False):
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
    st.info("📡 Haal dividend data op van Yahoo Finance (gratis & betrouwbaar). De data wordt eerst in een buffer opgeslagen zodat je deze kan controleren voordat je importeert.")

    # Haal portfolio aandelen op
    portfolio_stocks = get_portfolio_stocks_with_isin()

    if not portfolio_stocks:
        st.warning("Geen aandelen gevonden in je portfolio.")
    else:
        st.write(f"**{len(portfolio_stocks)} aandelen gevonden in je portfolio**")

        # Stap 1: Ophalen van API data
        st.write("### 📥 Stap 1: Data Ophalen van Yahoo Finance")

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            selected_stock = st.selectbox(
                "Selecteer aandeel om dividend data op te halen:",
                options=portfolio_stocks,
                format_func=lambda x: f"{x['name']} ({x['ticker']})"
            )

        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            debug_mode = st.checkbox("🔍 Debug", help="Toon raw data response")

        with col3:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔄 Haal Data Op", type="primary", use_container_width=True):
                with st.spinner(f"Dividenden ophalen voor {selected_stock['ticker']}..."):
                    result = fetch_dividends_from_yfinance(selected_stock['ticker'], debug=debug_mode)

                    if 'debug' in result:
                        # Debug mode - show raw response
                        st.write("**🔍 Debug Info:**")
                        st.json(result['debug'])

                    elif 'error' in result:
                        st.error(f"❌ {result['error']}")
                    elif 'dividends' in result:
                        dividends = result['dividends']
                        currency = result.get('currency', 'USD')

                        if not dividends:
                            st.info(f"Geen dividend geschiedenis gevonden voor {selected_stock['ticker']}")
                        else:
                            # Sla op in cache
                            cache_result = save_dividends_to_cache(
                                selected_stock['ticker'],
                                selected_stock['isin'],
                                dividends,
                                currency
                            )

                            st.success(f"✓ {len(dividends)} dividenden opgehaald en opgeslagen in buffer (currency: {currency})!")
                            st.info(f"ℹ️ Nieuw: {cache_result['added']}, Geüpdatet: {cache_result['updated']}")
                            st.rerun()

            # Toon laatste update per ticker
            if selected_stock:
                last_updated = get_cache_last_updated(selected_stock['ticker'])
                if last_updated:
                    last_updated_dt = datetime.fromisoformat(last_updated)
                    time_ago = datetime.now() - last_updated_dt

                    if time_ago.days > 0:
                        time_str = f"{time_ago.days} dagen geleden"
                    elif time_ago.seconds > 3600:
                        time_str = f"{time_ago.seconds // 3600} uur geleden"
                    else:
                        time_str = f"{time_ago.seconds // 60} minuten geleden"

                    st.caption(f"📅 Laatst opgehaald: {last_updated_dt.strftime('%Y-%m-%d %H:%M:%S')} ({time_str})")

            st.divider()

            # Stap 2: Bekijk en Importeer Buffer Data
            st.write("### 📋 Stap 2: Bekijk Buffer & Importeer")

            # Haal alle cached data op
            cached_data = get_cached_dividends()

            if cached_data.empty:
                st.info("Nog geen data in de buffer. Haal eerst data op via Stap 1.")
            else:
                # Toon snel import optie bovenaan
                unique_tickers = sorted(cached_data['ticker'].unique().tolist())

                st.success(f"✅ {len(cached_data)} dividenden klaar om te importeren van {len(unique_tickers)} aandelen")

                # Quick import all button
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("⚡ Importeer ALLE dividenden nu", type="primary", use_container_width=True, key="quick_import_all"):
                        imported_total = 0
                        skipped_total = 0

                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, ticker in enumerate(unique_tickers):
                            status_text.write(f"Importeren van {ticker}...")
                            progress_bar.progress((idx + 1) / len(unique_tickers))

                            result = import_all_from_cache_to_db(ticker)
                            imported_total += result['imported']
                            skipped_total += result['skipped']

                        progress_bar.empty()
                        status_text.empty()

                        if imported_total > 0:
                            st.success(f"✓ {imported_total} dividenden geïmporteerd!")
                        if skipped_total > 0:
                            st.info(f"ℹ️ {skipped_total} dividenden overgeslagen (al in database)")

                        st.rerun()

                st.divider()
                st.write("**Of selecteer specifiek aandeel:**")

                # Filter opties
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Default to first ticker instead of "Alle aandelen"
                    default_index = 1 if len(unique_tickers) > 0 else 0
                    filter_ticker = st.selectbox(
                        "Filter op aandeel:",
                        options=["Alle aandelen"] + unique_tickers,
                        key="filter_ticker",
                        index=default_index
                    )

                with col2:
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    if filter_ticker != "Alle aandelen":
                        if st.button("🗑️ Wis Buffer", type="secondary", use_container_width=True):
                            deleted = clear_cache(filter_ticker)
                            st.success(f"✓ {deleted} items verwijderd uit buffer")
                            st.rerun()

                # Filter data
                if filter_ticker != "Alle aandelen":
                    display_data = cached_data[cached_data['ticker'] == filter_ticker].copy()
                else:
                    display_data = cached_data.copy()

                # Format de data voor display
                display_data['fetched_at_formatted'] = pd.to_datetime(display_data['fetched_at']).dt.strftime('%Y-%m-%d %H:%M')
                display_data['amount_formatted'] = display_data.apply(
                    lambda row: format_currency(row['amount'], row.get('currency', 'USD')),
                    axis=1
                )

                st.dataframe(
                    display_data[['ticker', 'ex_date', 'currency', 'amount_formatted', 'fetched_at_formatted']].rename(columns={
                        'ticker': 'Ticker',
                        'ex_date': 'Ex-Dividend Datum',
                        'currency': 'Currency',
                        'amount_formatted': 'Bedrag',
                        'fetched_at_formatted': 'Opgehaald op'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.write(f"**{len(display_data)} dividenden in buffer**")

                # Import knoppen
                st.write("")  # Spacer

                if filter_ticker != "Alle aandelen":
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button(f"✅ Importeer {len(display_data)} dividenden van {filter_ticker}", type="primary", use_container_width=True, key="import_filtered"):
                            result = import_all_from_cache_to_db(filter_ticker)

                            if result['imported'] > 0:
                                st.success(f"✓ {result['imported']} dividenden geïmporteerd!")

                            if result['skipped'] > 0:
                                st.info(f"ℹ️ {result['skipped']} dividenden overgeslagen (al in database)")

                            if result['imported'] == 0 and result['skipped'] == 0:
                                st.warning("Geen dividenden om te importeren")

                            st.rerun()

                    with col2:
                        if st.button(f"🗑️ Wis {filter_ticker} uit buffer", type="secondary", use_container_width=True, key="clear_filtered"):
                            deleted = clear_cache(filter_ticker)
                            st.success(f"✓ {deleted} items verwijderd uit buffer")
                            st.rerun()
                else:
                    # Show option to import all from all tickers
                    st.warning("⚠️ Selecteer een specifiek aandeel om te importeren")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Wis Hele Buffer", type="secondary", use_container_width=True, key="clear_all"):
                        deleted = clear_cache()
                        st.success(f"✓ {deleted} items verwijderd uit buffer")
                        st.rerun()

            st.divider()
            st.info("""
            **Yahoo Finance voordelen:**
            - ✅ Gratis en geen rate limits
            - ✅ Uitstekende coverage voor US stocks
            - ✅ Goede coverage voor Europese stocks (met juiste ticker suffix)
            - ✅ Historische dividend data (vaak 10+ jaar)

            **Tips:**
            - Data wordt eerst in een buffer opgeslagen zodat je deze kan controleren
            - Gebruik de buffer om te voorkomen dat je dubbele data importeert
            - Voor Europese aandelen: zorg dat ticker de juiste suffix heeft (bijv. KBC.BR)
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
