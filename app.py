import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data.db")

# --------- Database helpers ---------
def get_connection():
    """Maakt (of opent) de SQLite database en zorgt dat de tabellen bestaan."""
    conn = sqlite3.connect(DB_PATH)
    
    # Transactions tabel
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            broker TEXT NOT NULL,
            transaction_type TEXT NOT NULL,
            name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            isin TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price_per_share REAL NOT NULL,
            currency TEXT NOT NULL,
            fees REAL NOT NULL,
            exchange_rate REAL DEFAULT 1.0,
            notes TEXT
        )
        """
    )
    
    # Cash balance tabel
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cash_balance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            broker TEXT NOT NULL,
            currency TEXT NOT NULL,
            amount REAL NOT NULL,
            last_updated TEXT NOT NULL,
            UNIQUE(broker, currency)
        )
        """
    )
    
    conn.commit()
    return conn

def insert_transaction(date, broker, transaction_type, name, ticker, isin, 
                      quantity, price_per_share, currency, fees, exchange_rate, notes):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO transactions 
        (date, broker, transaction_type, name, ticker, isin, quantity, 
         price_per_share, currency, fees, exchange_rate, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (date.isoformat(), broker, transaction_type, name, ticker.upper(), 
         isin.upper(), int(quantity), float(price_per_share), currency, 
         float(fees), float(exchange_rate), notes),
    )
    conn.commit()
    conn.close()

def load_transactions():
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT 
            id,
            date AS Datum, 
            broker AS Broker,
            transaction_type AS Type,
            name AS Naam, 
            ticker AS Ticker, 
            isin AS ISIN, 
            quantity AS Aantal,
            price_per_share AS Prijs,
            currency AS Valuta,
            fees AS Kosten,
            exchange_rate AS Wisselkoers,
            notes AS Notities
        FROM transactions 
        ORDER BY date DESC
        """,
        conn,
    )
    conn.close()
    return df

def delete_transaction(transaction_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
    conn.commit()
    conn.close()

# --------- Streamlit UI ---------
st.set_page_config(page_title="Portfolio Tracker", page_icon="📈", layout="wide")

st.title("📈 Portfolio Tracker")
st.subheader("Transactie invoeren")

# Formulier voor nieuwe transactie
with st.form("new_transaction"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date = st.date_input("Koopdatum", value=datetime.now())
        broker = st.selectbox(
            "Broker",
            ["DEGIRO", "Interactive Brokers", "Bolero", "Lynx", "Anders"]
        )
        if broker == "Anders":
            broker = st.text_input("Broker naam")
        
        transaction_type = st.radio(
            "Type transactie",
            ["KOOP", "VERKOOP"],
            horizontal=True
        )
    
    with col2:
        name = st.text_input("Naam aandeel")
        ticker = st.text_input("Ticker (bv. AAPL)")
        isin = st.text_input("ISIN")
        quantity = st.number_input(
            "Aantal aandelen",
            min_value=1,
            step=1,
            value=1
        )
    
    with col3:
        price_per_share = st.number_input(
            "Prijs per aandeel",
            min_value=0.0,
            step=0.01,
            format="%.4f",
        )
        currency = st.selectbox(
            "Valuta",
            ["EUR", "USD", "GBP", "CHF"]
        )
        fees = st.number_input(
            "Kosten (broker + taksen)",
            min_value=0.0,
            step=0.01,
            format="%.2f",
        )
        exchange_rate = st.number_input(
            "Wisselkoers naar EUR",
            min_value=0.0,
            step=0.0001,
            value=1.0 if currency == "EUR" else 0.0,
            format="%.4f",
            help="Laat op 1.0 als valuta EUR is"
        )
    
    notes = st.text_area("Notities (optioneel)", height=80)
    
    submitted = st.form_submit_button("💾 Opslaan", use_container_width=True)
    
    if submitted:
        if not (name and ticker and isin and quantity > 0 and price_per_share > 0):
            st.error("❌ Vul alle verplichte velden correct in.")
        else:
            insert_transaction(
                date, broker, transaction_type, name, ticker, isin,
                quantity, price_per_share, currency, fees, exchange_rate, notes
            )
            st.success("✅ Transactie opgeslagen!")
            st.rerun()

# Toon bestaande transacties
st.divider()
st.subheader("📋 Alle transacties")

try:
    df = load_transactions()
    if df.empty:
        st.info("Nog geen transacties opgeslagen.")
    else:
        # Voeg totaalbedrag kolom toe
        df['Totaal (€)'] = (df['Aantal'] * df['Prijs'] * df['Wisselkoers'] + df['Kosten']).round(2)
        
        # Filter opties
        col1, col2, col3 = st.columns(3)
        with col1:
            broker_filter = st.multiselect(
                "Filter op broker",
                options=df['Broker'].unique(),
                default=df['Broker'].unique()
            )
        with col2:
            type_filter = st.multiselect(
                "Filter op type",
                options=df['Type'].unique(),
                default=df['Type'].unique()
            )
        with col3:
            ticker_filter = st.multiselect(
                "Filter op ticker",
                options=df['Ticker'].unique(),
                default=df['Ticker'].unique()
            )
        
        # Gefilterde data
        filtered_df = df[
            (df['Broker'].isin(broker_filter)) &
            (df['Type'].isin(type_filter)) &
            (df['Ticker'].isin(ticker_filter))
        ]
        
        # Toon statistieken
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Aantal transacties", len(filtered_df))
        with col2:
            st.metric("Totaal geïnvesteerd", f"€{filtered_df['Totaal (€)'].sum():,.2f}")
        with col3:
            st.metric("Totale kosten", f"€{filtered_df['Kosten'].sum():,.2f}")
        with col4:
            unique_stocks = filtered_df['Ticker'].nunique()
            st.metric("Unieke aandelen", unique_stocks)
        
        st.divider()
        
        # Toon dataframe met verwijder optie
        display_df = filtered_df.copy()
        display_df = display_df.drop(columns=['id'])
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Verwijder transactie sectie
        with st.expander("🗑️ Transactie verwijderen"):
            transaction_to_delete = st.selectbox(
                "Selecteer transactie om te verwijderen",
                options=filtered_df['id'].tolist(),
                format_func=lambda x: f"ID {x}: {filtered_df[filtered_df['id']==x]['Datum'].values[0]} - {filtered_df[filtered_df['id']==x]['Ticker'].values[0]} ({filtered_df[filtered_df['id']==x]['Aantal'].values[0]} stuks)"
            )
            
            if st.button("🗑️ Verwijder geselecteerde transactie", type="secondary"):
                delete_transaction(transaction_to_delete)
                st.success("Transactie verwijderd!")
                st.rerun()
        
except Exception as e:
    st.error(f"⚠️ Fout bij laden van transacties: {e}")

# Footer
st.divider()
st.caption("💡 Tip: Gebruik de andere pagina's in de sidebar voor portfolio overzicht en grafieken.")