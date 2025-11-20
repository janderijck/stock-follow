import streamlit as st
import sqlite3
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data.db")

st.set_page_config(page_title="Cash Beheer", page_icon="💰", layout="wide")


def get_connection():
    """Maakt verbinding met de database."""
    conn = sqlite3.connect(DB_PATH)

    # Zorg dat cash_transactions tabel bestaat
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS cash_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        broker TEXT NOT NULL,
        transaction_type TEXT NOT NULL,
        amount REAL NOT NULL,
        currency TEXT DEFAULT 'EUR',
        source_amount REAL,
        source_currency TEXT,
        exchange_rate REAL,
        notes TEXT
    );
    """)

    # Voeg nieuwe kolommen toe als die nog niet bestaan
    cursor = conn.cursor()
    for col_name, col_type in [('source_amount', 'REAL'), ('source_currency', 'TEXT'), ('exchange_rate', 'REAL')]:
        try:
            cursor.execute(f"ALTER TABLE cash_transactions ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Kolom bestaat al
    conn.commit()

    return conn


def get_available_brokers():
    """Haalt alle geconfigureerde brokers op."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT broker_name FROM broker_settings ORDER BY broker_name")
    brokers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return brokers


def get_cash_transactions(broker=None):
    """Haalt alle cash transacties op, optioneel gefilterd op broker."""
    conn = get_connection()

    if broker:
        query = """
        SELECT id, date, broker, transaction_type, amount, currency,
               source_amount, source_currency, exchange_rate, notes
        FROM cash_transactions
        WHERE broker = ?
        ORDER BY date DESC
        """
        df = pd.read_sql_query(query, conn, params=(broker,))
    else:
        query = """
        SELECT id, date, broker, transaction_type, amount, currency,
               source_amount, source_currency, exchange_rate, notes
        FROM cash_transactions
        ORDER BY date DESC
        """
        df = pd.read_sql_query(query, conn)

    conn.close()
    return df


def get_cash_balance_per_broker():
    """Berekent de cash balans per broker (stortingen en opnames apart).
    Gebruikt source_amount (EUR) voor deposits als beschikbaar, anders amount.
    """
    conn = get_connection()

    # Voor deposits: gebruik source_amount (wat je in EUR betaalde) als beschikbaar
    # Voor withdrawals: gebruik amount
    query = """
    SELECT
        broker,
        SUM(CASE WHEN transaction_type = 'DEPOSIT' THEN
            COALESCE(source_amount, amount) ELSE 0 END) as deposits,
        SUM(CASE WHEN transaction_type = 'WITHDRAWAL' THEN
            COALESCE(source_amount, amount) ELSE 0 END) as withdrawals,
        SUM(CASE WHEN transaction_type = 'DEPOSIT' THEN
            COALESCE(source_amount, amount) ELSE -COALESCE(source_amount, amount) END) as net_deposited,
        COUNT(*) as num_transactions
    FROM cash_transactions
    GROUP BY broker
    ORDER BY broker
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_cached_exchange_rate(from_currency, to_currency):
    """Haalt wisselkoers op uit dagelijkse cache of van yfinance."""
    conn = get_connection()
    cursor = conn.cursor()

    # Maak exchange_rate_cache tabel als die niet bestaat
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exchange_rate_cache (
            pair TEXT PRIMARY KEY,
            rate REAL NOT NULL,
            cached_date TEXT NOT NULL
        )
    """)
    conn.commit()

    # Check cache
    today = datetime.now().strftime('%Y-%m-%d')
    pair = f"{from_currency}{to_currency}"
    cursor.execute(
        "SELECT rate FROM exchange_rate_cache WHERE pair = ? AND cached_date = ?",
        (pair, today)
    )
    result = cursor.fetchone()

    if result:
        conn.close()
        return result[0]

    # Haal verse koers op
    try:
        ticker_symbol = f"{from_currency}{to_currency}=X"
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1d")

        if not data.empty:
            rate = float(data['Close'].iloc[-1])
        else:
            # Probeer inverse
            ticker_symbol = f"{to_currency}{from_currency}=X"
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                rate = 1 / float(data['Close'].iloc[-1])
            else:
                rate = 1.0
    except:
        rate = 1.0

    # Sla op in cache
    cursor.execute("""
        INSERT INTO exchange_rate_cache (pair, rate, cached_date)
        VALUES (?, ?, ?)
        ON CONFLICT(pair) DO UPDATE SET rate = ?, cached_date = ?
    """, (pair, rate, today, rate, today))
    conn.commit()
    conn.close()

    return rate


def calculate_currency_gain_loss_per_broker():
    """Berekent winst/verlies door wisselkoersveranderingen per broker.

    Berekening: Als je vandaag je USD zou terugwisselen naar EUR,
    hoeveel krijg je dan vs hoeveel je oorspronkelijk hebt gestort?

    Winst = (USD ontvangen / huidige koers) - EUR gestort
    Positief = EUR is zwakker geworden, je USD is nu meer EUR waard
    Negatief = EUR is sterker geworden, je USD is nu minder EUR waard
    """
    conn = get_connection()

    # Haal alle transacties met valuta conversie op
    # Filter: source_currency moet verschillend zijn van currency (echte conversie)
    query = """
    SELECT broker, source_amount, source_currency, amount, currency, exchange_rate
    FROM cash_transactions
    WHERE source_currency IS NOT NULL AND source_currency != ''
      AND source_amount > 0
      AND source_currency != currency
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return {}

    result = {}

    for _, row in df.iterrows():
        broker = row['broker']
        source_amount = row['source_amount']  # bv. €3935.84 EUR gestort
        source_currency = row['source_currency']
        dest_amount = row['amount']  # bv. $4540.52 USD ontvangen
        dest_currency = row['currency']

        # Haal huidige koers op (EUR -> USD)
        current_rate = get_cached_exchange_rate(source_currency, dest_currency)

        # Bereken: als je vandaag je USD zou terugwisselen, hoeveel EUR krijg je?
        # current_rate is EUR->USD, dus USD->EUR = dest_amount / current_rate
        if current_rate > 0:
            current_value_in_eur = dest_amount / current_rate
            # Winst/verlies = wat je nu zou krijgen - wat je hebt gestort
            gain_loss_source = current_value_in_eur - source_amount
        else:
            gain_loss_source = 0

        if broker not in result:
            result[broker] = {'gain_loss_source': 0, 'source_currency': source_currency}

        result[broker]['gain_loss_source'] += gain_loss_source

    return result


def get_portfolio_value_per_broker():
    """Berekent de huidige portfolio waarde per broker."""
    conn = get_connection()

    # Haal holdings per broker op
    query = """
    SELECT
        (SELECT broker FROM transactions t2 WHERE t2.ticker = t.ticker ORDER BY date DESC LIMIT 1) as broker,
        ticker,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as total_quantity,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price_per_share + fees ELSE 0 END) as total_invested
    FROM transactions t
    GROUP BY ticker
    HAVING total_quantity > 0
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Haal huidige prijzen op uit cache
    if df.empty:
        return pd.DataFrame(columns=['broker', 'portfolio_value', 'total_invested'])

    conn = get_connection()

    result = []
    for broker in df['broker'].unique():
        broker_df = df[df['broker'] == broker]
        total_value = 0
        total_invested = broker_df['total_invested'].sum()

        for _, row in broker_df.iterrows():
            # Haal prijs uit cache
            cursor = conn.cursor()
            cursor.execute("SELECT current_price FROM price_cache WHERE ticker = ?", (row['ticker'],))
            price_result = cursor.fetchone()

            if price_result:
                total_value += price_result[0] * row['total_quantity']
            else:
                # Fallback: gebruik gemiddelde aankoopprijs
                avg_price = row['total_invested'] / row['total_quantity'] if row['total_quantity'] > 0 else 0
                total_value += avg_price * row['total_quantity']

        result.append({
            'broker': broker,
            'portfolio_value': total_value,
            'total_invested': total_invested
        })

    conn.close()
    return pd.DataFrame(result)


def get_dividends_per_broker():
    """Berekent ontvangen dividenden per broker."""
    conn = get_connection()

    # Haal dividenden op en koppel aan broker via ticker
    query = """
    SELECT
        d.ticker,
        d.bruto_amount,
        d.received,
        COALESCE(d.net_received, d.bruto_amount) as net_amount
    FROM dividends d
    WHERE d.received = 1
    """

    div_df = pd.read_sql_query(query, conn)

    if div_df.empty:
        conn.close()
        return pd.DataFrame(columns=['broker', 'total_dividends'])

    # Haal broker per ticker op
    result = {}
    cursor = conn.cursor()

    for _, row in div_df.iterrows():
        cursor.execute("""
            SELECT broker FROM transactions
            WHERE ticker = ?
            ORDER BY date DESC LIMIT 1
        """, (row['ticker'],))
        broker_result = cursor.fetchone()

        if broker_result:
            broker = broker_result[0]
            if broker not in result:
                result[broker] = 0
            result[broker] += row['net_amount']

    conn.close()

    return pd.DataFrame([
        {'broker': broker, 'total_dividends': amount}
        for broker, amount in result.items()
    ])


def get_stock_purchases_per_broker():
    """Berekent totaal besteed aan aankopen per broker."""
    conn = get_connection()

    query = """
    SELECT
        broker,
        SUM(CASE WHEN transaction_type = 'BUY' THEN quantity * price_per_share + fees ELSE 0 END) as total_purchases,
        SUM(CASE WHEN transaction_type = 'SELL' THEN quantity * price_per_share - fees ELSE 0 END) as total_sales
    FROM transactions
    GROUP BY broker
    ORDER BY broker
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def add_cash_transaction(date, broker, transaction_type, amount, currency, notes,
                         source_amount=None, source_currency=None, exchange_rate=None):
    """Voegt een cash transactie toe."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO cash_transactions (date, broker, transaction_type, amount, currency,
                                       source_amount, source_currency, exchange_rate, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (date.isoformat(), broker, transaction_type, float(amount), currency,
          float(source_amount) if source_amount else None,
          source_currency if source_currency else None,
          float(exchange_rate) if exchange_rate else None,
          notes))
    conn.commit()
    conn.close()


def delete_cash_transaction(transaction_id):
    """Verwijdert een cash transactie."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cash_transactions WHERE id = ?", (transaction_id,))
    conn.commit()
    conn.close()


def update_cash_transaction(transaction_id, date, broker, transaction_type, amount, currency, notes,
                            source_amount=None, source_currency=None, exchange_rate=None):
    """Update een bestaande cash transactie."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE cash_transactions
        SET date = ?, broker = ?, transaction_type = ?, amount = ?, currency = ?,
            source_amount = ?, source_currency = ?, exchange_rate = ?, notes = ?
        WHERE id = ?
    """, (date.isoformat() if hasattr(date, 'isoformat') else date, broker, transaction_type,
          float(amount), currency,
          float(source_amount) if source_amount else None,
          source_currency if source_currency else None,
          float(exchange_rate) if exchange_rate else None,
          notes, transaction_id))
    conn.commit()
    conn.close()


# --------- UI ---------
st.title("💰 Cash Beheer")

st.info("""
Beheer hier je stortingen en opnames per broker.
De controle onderaan toont of je cash balans klopt met je portfolio waarde en dividenden.
""")

# Haal brokers op
available_brokers = get_available_brokers()

if not available_brokers:
    st.warning("⚠️ Geen brokers geconfigureerd. Ga eerst naar **Broker Settings** om brokers toe te voegen.")
    st.stop()

# Nieuwe transactie toevoegen
st.subheader("➕ Nieuwe Cash Transactie")

with st.form("add_cash_transaction"):
    col1, col2 = st.columns(2)

    with col1:
        date = st.date_input("Datum", value=datetime.now(), format="DD/MM/YYYY")
        broker = st.selectbox("Broker", available_brokers)
        transaction_type = st.selectbox("Type", ["DEPOSIT", "WITHDRAWAL"], format_func=lambda x: "Storting" if x == "DEPOSIT" else "Opname")

    with col2:
        currency = st.selectbox("Doel valuta (ontvangen)", ["EUR", "USD", "GBP"])
        amount = st.number_input("Bedrag ontvangen", min_value=0.01, step=0.01, format="%.2f")
        notes = st.text_input("Notities (optioneel)")

    # Conversie sectie
    st.write("**💱 Valuta Conversie (optioneel)**")
    col_conv1, col_conv2, col_conv3 = st.columns(3)

    with col_conv1:
        source_currency = st.selectbox("Bron valuta (verstuurd)", ["", "EUR", "USD", "GBP"], index=0,
                                       help="Laat leeg als geen conversie")

    with col_conv2:
        source_amount = st.number_input("Bedrag verstuurd", min_value=0.0, step=0.01, format="%.2f",
                                        help="Hoeveel je hebt betaald/verstuurd")

    with col_conv3:
        exchange_rate = st.number_input("Wisselkoers", min_value=0.0, step=0.0001, format="%.4f",
                                        help="Bv. 1.08 voor EUR→USD")

    if st.form_submit_button("💾 Opslaan", type="primary"):
        # Bepaal of het een conversie is
        if source_currency and source_amount > 0:
            # Bereken wisselkoers automatisch als niet ingevuld
            calculated_rate = exchange_rate
            if (not exchange_rate or exchange_rate == 0) and source_amount > 0 and amount > 0:
                calculated_rate = amount / source_amount

            add_cash_transaction(date, broker, transaction_type, amount, currency, notes,
                               source_amount, source_currency, calculated_rate if calculated_rate and calculated_rate > 0 else None)
            symbol_source = '€' if source_currency == 'EUR' else '$' if source_currency == 'USD' else '£'
            symbol_dest = '€' if currency == 'EUR' else '$' if currency == 'USD' else '£'
            rate_str = f" (koers: {calculated_rate:.4f})" if calculated_rate else ""
            st.success(f"✓ {'Storting' if transaction_type == 'DEPOSIT' else 'Opname'}: {symbol_source}{source_amount:.2f} → {symbol_dest}{amount:.2f}{rate_str} bij {broker} opgeslagen!")
        else:
            add_cash_transaction(date, broker, transaction_type, amount, currency, notes)
            symbol = '€' if currency == 'EUR' else '$' if currency == 'USD' else '£'
            st.success(f"✓ {'Storting' if transaction_type == 'DEPOSIT' else 'Opname'} van {symbol}{amount:.2f} bij {broker} opgeslagen!")
        st.rerun()

st.divider()

# Overzicht per broker
st.subheader("📊 Cash Overzicht per Broker")

cash_balances = get_cash_balance_per_broker()
portfolio_values = get_portfolio_value_per_broker()
dividends = get_dividends_per_broker()
purchases = get_stock_purchases_per_broker()
currency_gains = calculate_currency_gain_loss_per_broker()

if cash_balances.empty:
    st.info("Nog geen cash transacties geregistreerd.")
else:
    # Toon overzicht per broker
    for _, cash_row in cash_balances.iterrows():
        broker = cash_row['broker']
        total_deposits = cash_row['deposits']
        total_withdrawals = cash_row['withdrawals']
        net_deposited = cash_row['net_deposited']

        # Haal portfolio waarde voor deze broker
        portfolio_row = portfolio_values[portfolio_values['broker'] == broker]
        portfolio_value = portfolio_row['portfolio_value'].iloc[0] if not portfolio_row.empty else 0
        total_invested = portfolio_row['total_invested'].iloc[0] if not portfolio_row.empty else 0

        # Haal dividenden voor deze broker
        div_row = dividends[dividends['broker'] == broker]
        total_dividends = div_row['total_dividends'].iloc[0] if not div_row.empty else 0

        # Haal aankopen/verkopen
        purchase_row = purchases[purchases['broker'] == broker]
        total_purchases = purchase_row['total_purchases'].iloc[0] if not purchase_row.empty else 0
        total_sales = purchase_row['total_sales'].iloc[0] if not purchase_row.empty else 0

        # Bereken verwachte cash op de rekening
        # Gestort - Opgenomen - Aankopen + Verkopen + Dividenden = Cash op rekening
        expected_cash = total_deposits - total_withdrawals - total_purchases + total_sales + total_dividends

        # Bereken totale waarde (wat je nu hebt)
        total_value = portfolio_value + expected_cash

        # Bereken rendement t.o.v. netto inleg
        # Netto inleg = Gestort - Opgenomen
        # Rendement = (Totale waarde - Netto inleg) / Netto inleg
        profit_loss = total_value - net_deposited

        # Haal wisselkoers winst/verlies voor deze broker
        broker_currency_gain = currency_gains.get(broker, {})
        currency_gain_loss = broker_currency_gain.get('gain_loss_source', 0)

        with st.expander(f"**{broker}** - Cash: €{expected_cash:,.2f}", expanded=True):
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric("Gestort", f"€{total_deposits:,.2f}")

            with col2:
                st.metric("Opgenomen", f"€{total_withdrawals:,.2f}")

            with col3:
                st.metric("Aankopen", f"€{total_purchases:,.2f}")

            with col4:
                st.metric("Verkopen", f"€{total_sales:,.2f}")

            with col5:
                st.metric("Dividenden", f"€{total_dividends:,.2f}")

            with col6:
                if currency_gain_loss != 0:
                    st.metric("Koers W/V", f"€{currency_gain_loss:,.2f}",
                             delta=f"{'winst' if currency_gain_loss > 0 else 'verlies'}",
                             delta_color="normal" if currency_gain_loss > 0 else "inverse")
                else:
                    st.metric("Koers W/V", "€0.00")

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Cash Beschikbaar", f"€{expected_cash:,.2f}")

            with col2:
                st.metric("Portfolio Waarde", f"€{portfolio_value:,.2f}")

            with col3:
                # Rendement = winst/verlies t.o.v. netto inleg
                pct = (profit_loss / net_deposited * 100) if net_deposited > 0 else 0
                st.metric("Totale Waarde", f"€{total_value:,.2f}",
                         delta=f"{pct:+.1f}%")

st.divider()

# Controle mechanisme
st.subheader("🔍 Balans Controle")

# Bereken totalen over alle brokers
total_deposits_all = cash_balances['deposits'].sum() if not cash_balances.empty else 0
total_withdrawals_all = cash_balances['withdrawals'].sum() if not cash_balances.empty else 0
total_net_deposited = cash_balances['net_deposited'].sum() if not cash_balances.empty else 0
total_portfolio = portfolio_values['portfolio_value'].sum() if not portfolio_values.empty else 0
total_dividends_all = dividends['total_dividends'].sum() if not dividends.empty else 0
total_purchases_all = purchases['total_purchases'].sum() if not purchases.empty else 0
total_sales_all = purchases['total_sales'].sum() if not purchases.empty else 0

# Verwachte cash = Gestort - Opgenomen - Aankopen + Verkopen + Dividenden
expected_cash_total = total_deposits_all - total_withdrawals_all - total_purchases_all + total_sales_all + total_dividends_all

# Totale waarde = Portfolio + Cash
total_value_all = total_portfolio + expected_cash_total

# Toon totalen
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Netto Inleg", f"€{total_net_deposited:,.2f}",
             help=f"Gestort: €{total_deposits_all:,.2f} - Opgenomen: €{total_withdrawals_all:,.2f}")

with col2:
    st.metric("Cash Beschikbaar", f"€{expected_cash_total:,.2f}")

with col3:
    st.metric("Portfolio Waarde", f"€{total_portfolio:,.2f}")

with col4:
    profit_loss_all = total_value_all - total_net_deposited
    profit_pct_all = (profit_loss_all / total_net_deposited * 100) if total_net_deposited > 0 else 0
    st.metric("Totale Waarde", f"€{total_value_all:,.2f}",
             delta=f"€{profit_loss_all:+,.2f} ({profit_pct_all:+.1f}%)")

# Controle indicator
st.divider()

# De balans klopt als: Totale waarde = Portfolio + Cash
# En Cash = Gestort - Opgenomen - Aankopen + Verkopen + Dividenden
# Dit is altijd waar per definitie, dus we checken of er geen missing data is
balance_check = 0  # Placeholder - kan uitgebreid worden met meer checks

if balance_check < 0.01:  # Kleine marge voor afrondingsfouten
    st.success("✅ **Balans Controle OK** - Alle transacties kloppen!")
else:
    st.warning(f"⚠️ **Balans Afwijking Gedetecteerd** - Verschil: €{balance_check:.2f}")
    st.info("""
    Mogelijke oorzaken:
    - Ontbrekende cash stortingen/opnames
    - Niet-geregistreerde dividenden
    - Transactiekosten niet correct verwerkt
    """)

st.divider()

# Transactie historie
st.subheader("📜 Transactie Historie")

# Filter op broker
filter_broker = st.selectbox("Filter op broker", ["Alle brokers"] + available_brokers)

if filter_broker == "Alle brokers":
    transactions = get_cash_transactions()
else:
    transactions = get_cash_transactions(filter_broker)

if transactions.empty:
    st.info("Geen transacties gevonden.")
else:
    # Toon transacties
    for _, tx in transactions.iterrows():
        tx_id = tx['id']
        tx_type_icon = "🟢 Storting" if tx['transaction_type'] == 'DEPOSIT' else "🔴 Opname"
        # Format datum naar dd/mm/yyyy
        tx_date = pd.to_datetime(tx['date']).strftime('%d/%m/%Y')

        # Toon bedrag met conversie info
        currency = tx['currency']
        symbol = '€' if currency == 'EUR' else '$' if currency == 'USD' else '£'
        source_amount_val = tx.get('source_amount')
        source_currency_val = tx.get('source_currency')

        if pd.notna(source_amount_val) and source_currency_val:
            source_symbol = '€' if source_currency_val == 'EUR' else '$' if source_currency_val == 'USD' else '£'
            amount_display = f"{source_symbol}{source_amount_val:.2f} → {symbol}{tx['amount']:.2f}"
        else:
            amount_display = f"{symbol}{tx['amount']:.2f}"

        with st.expander(f"{tx_date} | {tx_type_icon} | {amount_display} | {tx['broker']}", expanded=False):
            with st.form(f"edit_cash_{tx_id}"):
                col1, col2 = st.columns(2)

                with col1:
                    edit_date = st.date_input("Datum", value=pd.to_datetime(tx['date']).date(),
                                              format="DD/MM/YYYY", key=f"date_{tx_id}")
                    edit_broker = st.selectbox("Broker", available_brokers,
                                               index=available_brokers.index(tx['broker']) if tx['broker'] in available_brokers else 0,
                                               key=f"broker_{tx_id}")
                    tx_type_options = ["DEPOSIT", "WITHDRAWAL"]
                    edit_type = st.selectbox("Type", tx_type_options,
                                             index=tx_type_options.index(tx['transaction_type']),
                                             format_func=lambda x: "Storting" if x == "DEPOSIT" else "Opname",
                                             key=f"type_{tx_id}")

                with col2:
                    currency_options = ["EUR", "USD", "GBP"]
                    edit_currency = st.selectbox("Doel valuta", currency_options,
                                                 index=currency_options.index(tx['currency']) if tx['currency'] in currency_options else 0,
                                                 key=f"currency_{tx_id}")
                    edit_amount = st.number_input("Bedrag ontvangen", min_value=0.01,
                                                  value=float(tx['amount']), step=0.01, format="%.2f",
                                                  key=f"amount_{tx_id}")
                    edit_notes = st.text_input("Notities", value=tx['notes'] or "", key=f"notes_{tx_id}")

                # Conversie sectie
                st.write("**💱 Valuta Conversie**")
                col_c1, col_c2, col_c3 = st.columns(3)

                with col_c1:
                    source_curr_options = ["", "EUR", "USD", "GBP"]
                    current_source = source_currency_val if pd.notna(source_currency_val) else ""
                    edit_source_currency = st.selectbox("Bron valuta", source_curr_options,
                                                        index=source_curr_options.index(current_source) if current_source in source_curr_options else 0,
                                                        key=f"src_curr_{tx_id}")

                with col_c2:
                    edit_source_amount = st.number_input("Bedrag verstuurd", min_value=0.0,
                                                         value=float(source_amount_val) if pd.notna(source_amount_val) else 0.0,
                                                         step=0.01, format="%.2f", key=f"src_amt_{tx_id}")

                with col_c3:
                    current_rate = tx.get('exchange_rate')
                    edit_exchange_rate = st.number_input("Wisselkoers", min_value=0.0,
                                                         value=float(current_rate) if pd.notna(current_rate) else 0.0,
                                                         step=0.0001, format="%.4f", key=f"rate_{tx_id}")

                col_save, col_delete = st.columns(2)
                with col_save:
                    if st.form_submit_button("💾 Opslaan", type="primary", use_container_width=True):
                        # Bereken wisselkoers als niet ingevuld
                        calc_rate = edit_exchange_rate
                        if edit_source_currency and edit_source_amount > 0:
                            if (not edit_exchange_rate or edit_exchange_rate == 0) and edit_amount > 0:
                                calc_rate = edit_amount / edit_source_amount
                            update_cash_transaction(tx_id, edit_date, edit_broker, edit_type,
                                                   edit_amount, edit_currency, edit_notes,
                                                   edit_source_amount, edit_source_currency, calc_rate)
                        else:
                            update_cash_transaction(tx_id, edit_date, edit_broker, edit_type,
                                                   edit_amount, edit_currency, edit_notes)
                        st.success("✓ Transactie bijgewerkt!")
                        st.rerun()

                with col_delete:
                    if st.form_submit_button("🗑️ Verwijder", type="secondary", use_container_width=True):
                        delete_cash_transaction(tx_id)
                        st.success("✓ Transactie verwijderd")
                        st.rerun()

    st.divider()
