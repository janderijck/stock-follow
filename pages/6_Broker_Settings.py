import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data.db")

st.set_page_config(page_title="Broker Settings", page_icon="⚙️", layout="wide")


# --------- Database helpers ---------
def get_connection():
    """Maakt verbinding met de database."""
    conn = sqlite3.connect(DB_PATH)

    # Maak broker_settings tabel
    conn.execute("""
        CREATE TABLE IF NOT EXISTS broker_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            broker_name TEXT NOT NULL UNIQUE,
            country TEXT NOT NULL,
            has_w8ben INTEGER DEFAULT 0,
            w8ben_expiry_date TEXT,
            historical_costs REAL DEFAULT 0,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Voeg historical_costs kolom toe als die nog niet bestaat
    try:
        conn.execute("ALTER TABLE broker_settings ADD COLUMN historical_costs REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Kolom bestaat al

    # Maak stock_info tabel voor asset type tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            isin TEXT,
            name TEXT,
            asset_type TEXT DEFAULT 'STOCK',
            country TEXT,
            yahoo_ticker TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Voeg yahoo_ticker kolom toe als die nog niet bestaat
    try:
        conn.execute("ALTER TABLE stock_info ADD COLUMN yahoo_ticker TEXT")
    except sqlite3.OperationalError:
        pass  # Kolom bestaat al

    # Maak api_settings tabel voor API keys
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_settings (
            id INTEGER PRIMARY KEY,
            setting_key TEXT NOT NULL UNIQUE,
            setting_value TEXT,
            updated_at TEXT
        )
    """)

    # Maak tax_settings tabel voor algemene belasting settings
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tax_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_name TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            description TEXT,
            updated_at TEXT NOT NULL
        )
    """)

    conn.commit()
    return conn


def initialize_default_tax_settings():
    """Initialiseer default tax rates."""
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    default_settings = [
        ('belgian_dividend_tax', '0.30', 'Belgische roerende voorheffing op dividenden'),
        ('belgian_reit_tax', '0.489', 'Belgische REIT belasting (GVV/GVBF zoals Aedifica)'),
        ('us_dividend_tax_with_w8', '0.15', 'Amerikaanse bronheffing met W-8BEN'),
        ('us_dividend_tax_without_w8', '0.30', 'Amerikaanse bronheffing zonder W-8BEN'),
    ]

    for name, value, description in default_settings:
        cursor.execute("""
            INSERT OR IGNORE INTO tax_settings (setting_name, value, description, updated_at)
            VALUES (?, ?, ?, ?)
        """, (name, value, description, now))

    conn.commit()
    conn.close()


def get_tax_setting(name, default=None):
    """Haal een tax setting op."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM tax_settings WHERE setting_name = ?", (name,))
    result = cursor.fetchone()
    conn.close()
    return float(result[0]) if result else default


def update_tax_setting(name, value):
    """Update een tax setting."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        UPDATE tax_settings
        SET value = ?, updated_at = ?
        WHERE setting_name = ?
    """, (str(value), now, name))
    conn.commit()
    conn.close()


def get_all_brokers():
    """Haal alle broker configuraties op."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT id, broker_name, country, has_w8ben, w8ben_expiry_date, historical_costs, notes, updated_at
        FROM broker_settings
        ORDER BY broker_name
    """, conn)
    conn.close()
    return df


def add_broker(broker_name, country, has_w8ben, w8ben_expiry_date=None, historical_costs=0, notes=""):
    """Voeg een nieuwe broker toe."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    try:
        cursor.execute("""
            INSERT INTO broker_settings (broker_name, country, has_w8ben, w8ben_expiry_date, historical_costs, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (broker_name, country, 1 if has_w8ben else 0, w8ben_expiry_date, float(historical_costs), notes, now, now))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def update_broker(broker_id, broker_name, country, has_w8ben, w8ben_expiry_date=None, historical_costs=0, notes=""):
    """Update een bestaande broker."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    cursor.execute("""
        UPDATE broker_settings
        SET broker_name = ?, country = ?, has_w8ben = ?, w8ben_expiry_date = ?, historical_costs = ?, notes = ?, updated_at = ?
        WHERE id = ?
    """, (broker_name, country, 1 if has_w8ben else 0, w8ben_expiry_date, float(historical_costs), notes, now, broker_id))

    conn.commit()
    conn.close()


def delete_broker(broker_id):
    """Verwijder een broker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM broker_settings WHERE id = ?", (broker_id,))
    conn.commit()
    conn.close()


def get_api_setting(key):
    """Haal een API setting op."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT setting_value FROM api_settings WHERE setting_key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def set_api_setting(key, value):
    """Sla een API setting op."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO api_settings (setting_key, setting_value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(setting_key) DO UPDATE SET
            setting_value = ?, updated_at = ?
    """, (key, value, now, value, now))
    conn.commit()
    conn.close()


def get_manual_prices():
    """Haal alle handmatig ingestelde prijzen op."""
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
    conn.commit()

    df = pd.read_sql_query("SELECT * FROM manual_prices ORDER BY ticker", conn)
    conn.close()
    return df


def set_manual_price(ticker, price, currency='EUR'):
    """Stel een handmatige prijs in voor een ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO manual_prices (ticker, price, currency, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            price = ?, currency = ?, updated_at = ?
    """, (ticker, price, currency, now, price, currency, now))

    conn.commit()
    conn.close()


def delete_manual_price(ticker):
    """Verwijder een handmatige prijs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM manual_prices WHERE ticker = ?", (ticker,))
    conn.commit()
    conn.close()


def get_portfolio_tickers():
    """Haal alle tickers op uit de portfolio."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT ticker, name FROM transactions ORDER BY name
    """)
    result = cursor.fetchall()
    conn.close()
    return result


def get_all_stocks():
    """Haal alle stock configuraties op."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT si.id, si.ticker, si.isin, si.name, si.asset_type, si.country,
               si.custom_dividend_tax_rate, si.yahoo_ticker, si.notes
        FROM stock_info si
        ORDER BY si.ticker
    """, conn)
    conn.close()
    return df


def add_or_update_stock(ticker, isin, name, asset_type, country, notes="", custom_tax_rate=None, yahoo_ticker=None):
    """Voeg een stock toe of update bestaande."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    # Check of bestaat
    cursor.execute("SELECT id FROM stock_info WHERE ticker = ?", (ticker,))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE stock_info
            SET isin = ?, name = ?, asset_type = ?, country = ?,
                custom_dividend_tax_rate = ?, yahoo_ticker = ?, notes = ?, updated_at = ?
            WHERE ticker = ?
        """, (isin, name, asset_type, country, custom_tax_rate, yahoo_ticker, notes, now, ticker))
    else:
        cursor.execute("""
            INSERT INTO stock_info (ticker, isin, name, asset_type, country,
                                   custom_dividend_tax_rate, yahoo_ticker, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, isin, name, asset_type, country, custom_tax_rate, yahoo_ticker, notes, now, now))

    conn.commit()
    conn.close()


def delete_stock(stock_id):
    """Verwijder een stock configuratie."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM stock_info WHERE id = ?", (stock_id,))
    conn.commit()
    conn.close()


def get_portfolio_stocks():
    """Haal alle unieke stocks uit de portfolio."""
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT DISTINCT ticker, name, isin
        FROM transactions
        ORDER BY ticker
    """, conn)
    conn.close()
    return df


# Initialize default settings
initialize_default_tax_settings()


# --------- UI ---------
st.title("⚙️ Broker & Tax Settings")
st.write("Configureer je broker settings en belastingtarieven voor accurate dividend berekeningen")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏦 Broker Settings", "📊 Stock/Asset Types", "💰 Tax Rates", "🔑 API Settings", "💵 Handmatige Prijzen"])

# Tab 1: Broker Settings
with tab1:
    st.subheader("Broker Configuratie")
    st.info("Configureer je brokers en W-8BEN status voor accurate Amerikaanse bronheffing berekeningen")

    # Add/Edit broker form
    st.write("### Broker Toevoegen/Bewerken")

    with st.form("broker_form"):
        col1, col2 = st.columns(2)

        with col1:
            broker_name = st.text_input("Broker naam", placeholder="DeGiro, Interactive Brokers, etc.")
            country = st.selectbox("Land", ["België", "Nederland", "Verenigde Staten", "Verenigd Koninkrijk", "Andere"])
            has_w8ben = st.checkbox("W-8BEN formulier actief", help="Voor Amerikaanse dividenden: 15% bronheffing met W-8BEN, 30% zonder")

        with col2:
            w8ben_expiry = None
            if has_w8ben:
                w8ben_expiry = st.date_input("W-8BEN vervaldatum", help="W-8BEN is normaal 3 jaar geldig", format="DD/MM/YYYY")
            historical_costs = st.number_input(
                "Historische kosten (€)",
                min_value=0.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                help="Transactiekosten van voor de app, om mee te rekenen in cash balans"
            )
            notes = st.text_area("Notities (optioneel)")

        submit = st.form_submit_button("💾 Broker Opslaan", type="primary")

        if submit:
            if broker_name:
                expiry_str = w8ben_expiry.isoformat() if w8ben_expiry else None
                if add_broker(broker_name, country, has_w8ben, expiry_str, historical_costs, notes):
                    st.success(f"✓ Broker '{broker_name}' toegevoegd!")
                    st.rerun()
                else:
                    st.error(f"❌ Broker '{broker_name}' bestaat al. Gebruik de edit functie hieronder.")
            else:
                st.error("Voer een broker naam in")

    st.divider()

    # Display existing brokers
    st.write("### Bestaande Brokers")

    brokers_df = get_all_brokers()

    if brokers_df.empty:
        st.info("Nog geen brokers geconfigureerd. Voeg er hierboven een toe!")
    else:
        for idx, row in brokers_df.iterrows():
            broker_id = row['id']
            with st.expander(f"🏦 {row['broker_name']} ({row['country']})", expanded=False):
                with st.form(f"edit_broker_{broker_id}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        edit_broker_name = st.text_input("Broker naam", value=row['broker_name'], key=f"name_{broker_id}")
                        country_options = ["België", "Nederland", "Verenigde Staten", "Verenigd Koninkrijk", "Andere"]
                        current_country = row['country']
                        country_idx = country_options.index(current_country) if current_country in country_options else 4
                        edit_country = st.selectbox("Land", country_options, index=country_idx, key=f"country_{broker_id}")
                        edit_w8ben = st.checkbox("W-8BEN formulier actief", value=bool(row['has_w8ben']), key=f"w8ben_{broker_id}")

                    with col2:
                        if row['w8ben_expiry_date']:
                            expiry_value = pd.to_datetime(row['w8ben_expiry_date']).date()
                        else:
                            expiry_value = datetime.now().date()
                        edit_w8ben_expiry = st.date_input("W-8BEN vervaldatum", value=expiry_value, key=f"expiry_{broker_id}", format="DD/MM/YYYY")

                        hist_costs = row.get('historical_costs', 0) or 0
                        edit_hist_costs = st.number_input(
                            "Historische kosten (€)",
                            min_value=0.0,
                            value=float(hist_costs),
                            step=0.01,
                            format="%.2f",
                            key=f"hist_{broker_id}",
                            help="Transactiekosten van voor de app"
                        )
                        edit_notes = st.text_area("Notities", value=row['notes'] or "", key=f"notes_{broker_id}", height=68)

                    col_save, col_delete = st.columns(2)
                    with col_save:
                        if st.form_submit_button("💾 Opslaan", type="primary", use_container_width=True):
                            expiry_str = edit_w8ben_expiry.isoformat() if edit_w8ben else None
                            update_broker(broker_id, edit_broker_name, edit_country, edit_w8ben, expiry_str, edit_hist_costs, edit_notes)
                            st.success("✓ Broker bijgewerkt!")
                            st.rerun()
                    with col_delete:
                        if st.form_submit_button("🗑️ Verwijder", type="secondary", use_container_width=True):
                            delete_broker(broker_id)
                            st.success("✓ Broker verwijderd")
                            st.rerun()

# Tab 2: Stock/Asset Types
with tab2:
    st.subheader("Stock/Asset Type Configuratie")
    st.info("Markeer welke aandelen REITs zijn (zoals Aedifica) voor correcte belastingberekening (48.9% vs 30%)")

    # Show portfolio stocks
    portfolio_stocks = get_portfolio_stocks()

    if portfolio_stocks.empty:
        st.warning("Geen aandelen gevonden in je portfolio")
    else:
        st.write(f"**{len(portfolio_stocks)} aandelen in je portfolio**")

        # Get existing stock info
        stock_info_df = get_all_stocks()

        # Merge with portfolio
        if not stock_info_df.empty:
            portfolio_stocks = portfolio_stocks.merge(
                stock_info_df[['ticker', 'asset_type', 'country', 'custom_dividend_tax_rate']],
                on='ticker',
                how='left'
            )
        else:
            portfolio_stocks['asset_type'] = 'STOCK'
            portfolio_stocks['country'] = None
            portfolio_stocks['custom_dividend_tax_rate'] = None

        # Fill NaN values with defaults
        portfolio_stocks['asset_type'] = portfolio_stocks['asset_type'].fillna('STOCK')
        # Country can stay as NaN, we'll handle it in the form

        st.write("### Configureer Asset Types")

        # Form voor bulk edit
        for idx, stock in portfolio_stocks.iterrows():
            with st.expander(f"📊 {stock['name']} ({stock['ticker']})", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    asset_type = st.selectbox(
                        "Asset Type",
                        ["STOCK", "REIT", "ETF"],
                        index=["STOCK", "REIT", "ETF"].index(stock.get('asset_type', 'STOCK')),
                        key=f"asset_type_{stock['ticker']}"
                    )

                    if asset_type == "REIT":
                        st.warning("⚠️ REIT: Belgische belasting van 48.9%")

                with col2:
                    # Handle country selection with NaN/None values
                    country_options = ["België", "Verenigde Staten", "Nederland", "Frankrijk", "Duitsland", "Verenigd Koninkrijk", "Andere"]
                    current_country = stock.get('country')

                    # Determine index: 0 if NaN/None, otherwise find in list
                    if pd.isna(current_country) or not current_country:
                        country_index = 0
                    elif current_country in country_options:
                        country_index = country_options.index(current_country)
                    else:
                        country_index = 6  # "Andere"

                    stock_country = st.selectbox(
                        "Land van uitgifte",
                        country_options,
                        index=country_index,
                        key=f"country_{stock['ticker']}"
                    )

                # Custom dividend tax rate input
                st.write("**Aangepast dividend belastingtarief (optioneel)**")
                col_tax1, col_tax2 = st.columns([2, 3])

                with col_tax1:
                    # Get existing custom tax rate if available
                    existing_custom_rate = stock.get('custom_dividend_tax_rate')
                    default_value = (existing_custom_rate * 100) if pd.notna(existing_custom_rate) else None

                    custom_tax_pct = st.number_input(
                        "Percentage (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=default_value,
                        step=0.1,
                        format="%.1f",
                        key=f"custom_tax_{stock['ticker']}",
                        help="Aangepast tarief ALLEEN voor dividendbelasting. Bijvoorbeeld: Aedifica heeft 15% i.p.v. 48.9% wegens residentiële gezondheidszorg REIT."
                    )

                with col_tax2:
                    if custom_tax_pct is not None:
                        st.info(f"ℹ️ Gebruikt {custom_tax_pct}% i.p.v. standaard tarief")
                    elif asset_type == "REIT":
                        st.caption("Standaard: 48.9%")
                    else:
                        st.caption("Standaard: 30%")

                stock_notes = st.text_input(
                    "Notities",
                    value="",
                    key=f"notes_{stock['ticker']}"
                )

                with col3:
                    st.write("")
                    st.write("")
                    if st.button("💾 Opslaan", key=f"save_{stock['ticker']}", type="primary"):
                        # Convert percentage to decimal
                        custom_tax_rate = custom_tax_pct / 100.0 if custom_tax_pct is not None else None

                        add_or_update_stock(
                            stock['ticker'],
                            stock['isin'],
                            stock['name'],
                            asset_type,
                            stock_country,
                            stock_notes,
                            custom_tax_rate
                        )
                        st.success(f"✓ {stock['ticker']} opgeslagen")
                        st.rerun()

        st.divider()

        # Show summary
        st.write("### Overzicht")

        configured_stocks = get_all_stocks()

        if not configured_stocks.empty:
            # Summary stats
            col1, col2, col3 = st.columns(3)

            with col1:
                regular_count = len(configured_stocks[configured_stocks['asset_type'] == 'STOCK'])
                st.metric("Reguliere Aandelen", regular_count)

            with col2:
                reit_count = len(configured_stocks[configured_stocks['asset_type'] == 'REIT'])
                st.metric("REITs", reit_count)

            with col3:
                etf_count = len(configured_stocks[configured_stocks['asset_type'] == 'ETF'])
                st.metric("ETFs", etf_count)

            st.divider()

            # Display tabel with custom tax rate
            display_df = configured_stocks[['ticker', 'name', 'asset_type', 'country', 'custom_dividend_tax_rate']].copy()

            # Format custom tax rate as percentage
            display_df['custom_dividend_tax_rate'] = display_df['custom_dividend_tax_rate'].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"
            )

            display_df.columns = ['Ticker', 'Naam', 'Type', 'Land', 'Aangepast Dividend Tarief']
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# Tab 3: Tax Rates
with tab3:
    st.subheader("Belasting Tarieven")
    st.info("Configureer de belastingtarieven voor verschillende jurisdicties")

    # Display and edit tax settings
    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🇧🇪 Belgische Belastingen")

        belgian_div_tax = get_tax_setting('belgian_dividend_tax', 0.30)
        belgian_reit_tax = get_tax_setting('belgian_reit_tax', 0.489)

        with st.form("belgian_tax_form"):
            new_belgian_div = st.number_input(
                "Roerende voorheffing dividenden (%)",
                min_value=0.0,
                max_value=100.0,
                value=belgian_div_tax * 100,
                step=0.1,
                format="%.1f"
            )

            new_belgian_reit = st.number_input(
                "REIT belasting (%) - GVV/GVBF",
                min_value=0.0,
                max_value=100.0,
                value=belgian_reit_tax * 100,
                step=0.1,
                format="%.1f",
                help="Bijvoorbeeld: Aedifica, WDP, Xior, etc."
            )

            if st.form_submit_button("💾 Opslaan", type="primary"):
                update_tax_setting('belgian_dividend_tax', new_belgian_div / 100)
                update_tax_setting('belgian_reit_tax', new_belgian_reit / 100)
                st.success("✓ Belgische belastingtarieven bijgewerkt")
                st.rerun()

        st.divider()

        st.write("**Huidige tarieven:**")
        st.write(f"- Dividenden: {belgian_div_tax * 100:.1f}%")
        st.write(f"- REITs: {belgian_reit_tax * 100:.1f}%")

    with col2:
        st.write("### 🇺🇸 Amerikaanse Bronheffing")

        us_with_w8 = get_tax_setting('us_dividend_tax_with_w8', 0.15)
        us_without_w8 = get_tax_setting('us_dividend_tax_without_w8', 0.30)

        with st.form("us_tax_form"):
            new_us_with = st.number_input(
                "Met W-8BEN formulier (%)",
                min_value=0.0,
                max_value=100.0,
                value=us_with_w8 * 100,
                step=0.1,
                format="%.1f"
            )

            new_us_without = st.number_input(
                "Zonder W-8BEN formulier (%)",
                min_value=0.0,
                max_value=100.0,
                value=us_without_w8 * 100,
                step=0.1,
                format="%.1f"
            )

            if st.form_submit_button("💾 Opslaan", type="primary"):
                update_tax_setting('us_dividend_tax_with_w8', new_us_with / 100)
                update_tax_setting('us_dividend_tax_without_w8', new_us_without / 100)
                st.success("✓ Amerikaanse belastingtarieven bijgewerkt")
                st.rerun()

        st.divider()

        st.write("**Huidige tarieven:**")
        st.write(f"- Met W-8BEN: {us_with_w8 * 100:.1f}%")
        st.write(f"- Zonder W-8BEN: {us_without_w8 * 100:.1f}%")

# Tab 4: API Settings
with tab4:
    st.subheader("API Configuratie")
    st.write("Configureer API keys voor externe koersdatabronnen.")

    st.info("""
    **Hoe het werkt:**
    1. Eerst wordt Yahoo Finance geprobeerd (gratis, geen key nodig)
    2. Als Yahoo faalt, wordt Marketstack gebruikt (als key ingesteld)

    Marketstack heeft betere Europese dekking en kan zoeken op ISIN.
    """)

    st.divider()

    st.write("### Marketstack API")
    st.write("Krijg een gratis API key op: https://marketstack.com/signup/free")

    current_key = get_api_setting('marketstack_api_key') or ''

    with st.form("marketstack_form"):
        api_key = st.text_input(
            "Marketstack API Key",
            value=current_key,
            type="password",
            help="Je Marketstack API key (gratis tier: 100 calls/maand)"
        )

        if st.form_submit_button("💾 Opslaan", type="primary"):
            set_api_setting('marketstack_api_key', api_key.strip())
            st.success("✓ API key opgeslagen")
            st.rerun()

    if current_key:
        st.success("✅ Marketstack API key is geconfigureerd")
    else:
        st.warning("⚠️ Geen Marketstack API key ingesteld - alleen Yahoo Finance wordt gebruikt")

# Tab 5: Handmatige Prijzen
with tab5:
    st.subheader("Handmatige Prijzen")
    st.write("Stel handmatig prijzen in voor aandelen die niet correct worden opgehaald via API's.")

    st.info("""
    **Wanneer te gebruiken:**
    - ETFs die niet op Yahoo Finance staan in EUR
    - Aandelen met verkeerde prijzen
    - Belgische/Nederlandse aandelen zonder API dekking

    Handmatige prijzen hebben **prioriteit** boven API prijzen.
    """)

    st.divider()

    # Nieuwe prijs toevoegen
    st.write("### Nieuwe Prijs Instellen")

    portfolio_tickers = get_portfolio_tickers()

    if portfolio_tickers:
        with st.form("add_manual_price"):
            col1, col2, col3 = st.columns(3)

            with col1:
                ticker_options = [f"{t[0]} - {t[1]}" for t in portfolio_tickers]
                selected = st.selectbox("Aandeel", ticker_options)
                selected_ticker = selected.split(" - ")[0] if selected else ""

            with col2:
                price = st.number_input("Prijs", min_value=0.0, step=0.01, format="%.2f")

            with col3:
                currency = st.selectbox("Valuta", ["EUR", "USD", "GBP"])

            if st.form_submit_button("💾 Opslaan", type="primary"):
                if selected_ticker and price > 0:
                    set_manual_price(selected_ticker, price, currency)
                    st.success(f"✓ Prijs voor {selected_ticker} ingesteld op {currency} {price:.2f}")
                    st.rerun()
                else:
                    st.error("Selecteer een aandeel en vul een prijs in")
    else:
        st.warning("Geen aandelen in portfolio. Voeg eerst transacties toe.")

    st.divider()

    # Bestaande handmatige prijzen
    st.write("### Ingestelde Prijzen")

    manual_prices = get_manual_prices()

    if manual_prices.empty:
        st.info("Nog geen handmatige prijzen ingesteld.")
    else:
        for _, row in manual_prices.iterrows():
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.write(f"**{row['ticker']}**")

            with col2:
                symbol = '€' if row['currency'] == 'EUR' else '$' if row['currency'] == 'USD' else '£'
                st.write(f"{symbol}{row['price']:.2f}")

            with col3:
                if st.button("🗑️", key=f"del_manual_{row['ticker']}", help="Verwijder"):
                    delete_manual_price(row['ticker'])
                    st.rerun()

            # Toon laatste update
            if row['updated_at']:
                updated = pd.to_datetime(row['updated_at']).strftime('%d/%m/%Y %H:%M')
                st.caption(f"Laatst bijgewerkt: {updated}")

            st.divider()

st.divider()

# Info sectie
with st.expander("ℹ️ Belasting Informatie"):
    st.write("""
    ### Belgische Dividend Belasting

    **Reguliere aandelen:** 30% roerende voorheffing
    - Dit is de standaard belasting op dividenden in België
    - Deze wordt automatisch ingehouden door je broker

    **REITs (GVV/GVBF):** 48.9% belasting
    - Belgische REITs zoals Aedifica, WDP, Xior hebben een hoger tarief
    - Dit komt door de specifieke fiscale status van GVV/GVBF

    ### Amerikaanse Bronheffing

    **Met W-8BEN formulier:** 15% bronheffing
    - Verdrag tussen België en VS reduceert bronheffing naar 15%
    - W-8BEN moet ingevuld zijn bij je broker
    - Geldig voor 3 jaar, daarna opnieuw indienen

    **Zonder W-8BEN:** 30% bronheffing
    - Standaard Amerikaanse bronheffing
    - Kan teruggevorderd worden via belastingaangifte (complex)

    ### Dubbele Belasting

    Voor Amerikaanse dividenden betaal je:
    1. Amerikaanse bronheffing (15% of 30%)
    2. Belgische roerende voorheffing (30%)

    De Amerikaanse bronheffing kan verrekend worden in je Belgische aangifte.

    **Voorbeeld met W-8BEN:**
    - Bruto dividend: $100
    - Amerikaanse bronheffing (15%): -$15
    - Netto ontvangen: $85
    - Belgische RV (30% van bruto): -$30
    - Verrekening: +$15 (in aangifte)
    - Uiteindelijk netto: $85 (= 70% van bruto)

    **Belangrijke tips:**
    - Vul altijd je W-8BEN in bij Amerikaanse brokers
    - Hou bij hoeveel Amerikaanse bronheffing je hebt betaald
    - Gebruik deze info in je belastingaangifte voor verrekening
    """)
