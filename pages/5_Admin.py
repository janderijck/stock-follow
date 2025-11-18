import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data.db")

st.set_page_config(page_title="Admin - Database Beheer", page_icon="⚙️", layout="wide")


# --------- Database helpers ---------
def get_connection():
    """Maakt verbinding met de database."""
    return sqlite3.connect(DB_PATH)


def get_table_names():
    """Haalt alle tabelnamen op."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def get_table_data(table_name):
    """Haalt alle data op van een tabel."""
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def get_table_schema(table_name):
    """Haalt het schema op van een tabel."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    schema = cursor.fetchall()
    conn.close()
    return schema


def delete_record(table_name, id_value):
    """Verwijdert een record uit een tabel op basis van ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table_name} WHERE id = ?", (id_value,))
    conn.commit()
    conn.close()


def execute_custom_query(query):
    """Voert een custom SQL query uit (read-only)."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        conn.close()
        return None, str(e)


# --------- UI ---------
st.title("⚙️ Admin - Database Beheer")
st.warning("⚠️ **Let op:** Dit is een admin pagina. Gebruik voorzichtig!")

# Tabbladen voor verschillende functies
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tabel Data", "🔍 Custom Query", "📝 Schema Info", "📈 Database Stats"])

# Tab 1: Tabel Data
with tab1:
    st.subheader("Database Tabellen")

    tables = get_table_names()

    if not tables:
        st.info("Geen tabellen gevonden in de database.")
    else:
        selected_table = st.selectbox("Selecteer een tabel:", tables)

        if selected_table:
            st.write(f"### Tabel: `{selected_table}`")

            # Haal data op
            df = get_table_data(selected_table)

            if df.empty:
                st.info(f"Tabel `{selected_table}` is leeg.")
            else:
                st.write(f"**Aantal rijen:** {len(df)}")

                # Toon data
                st.dataframe(df, use_container_width=True, hide_index=False)

                # Export functionaliteit
                st.download_button(
                    label="📥 Download als CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{selected_table}.csv",
                    mime="text/csv"
                )

                # Delete functionaliteit (alleen voor tabellen met 'id' kolom)
                if 'id' in df.columns:
                    st.divider()
                    with st.expander("🗑️ Record verwijderen (gevaarlijk!)", expanded=False):
                        st.warning("⚠️ Dit kan niet ongedaan gemaakt worden!")

                        delete_id = st.number_input(
                            "ID van record om te verwijderen:",
                            min_value=1,
                            step=1,
                            key=f"delete_{selected_table}"
                        )

                        if st.button(f"🗑️ Verwijder record met ID {delete_id}", type="secondary"):
                            if delete_id in df['id'].values:
                                delete_record(selected_table, delete_id)
                                st.success(f"Record met ID {delete_id} verwijderd uit {selected_table}")
                                st.rerun()
                            else:
                                st.error(f"ID {delete_id} bestaat niet in {selected_table}")

# Tab 2: Custom Query
with tab2:
    st.subheader("Custom SQL Query (Read-Only)")
    st.info("💡 Gebruik deze functie om custom SELECT queries uit te voeren op de database.")

    query = st.text_area(
        "SQL Query:",
        height=150,
        placeholder="SELECT * FROM transactions WHERE ticker = 'AAPL' LIMIT 10",
        key="custom_query"
    )

    if st.button("▶️ Voer Query Uit", type="primary"):
        if query.strip():
            # Simpele veiligheidscheck (alleen SELECT)
            if not query.strip().upper().startswith('SELECT'):
                st.error("❌ Alleen SELECT queries zijn toegestaan voor veiligheid!")
            else:
                result_df, error = execute_custom_query(query)

                if error:
                    st.error(f"❌ Query fout: {error}")
                elif result_df is not None:
                    st.success(f"✓ Query succesvol! ({len(result_df)} rijen)")
                    st.dataframe(result_df, use_container_width=True)

                    # Export
                    st.download_button(
                        label="📥 Download resultaat als CSV",
                        data=result_df.to_csv(index=False).encode('utf-8'),
                        file_name="query_result.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("Voer eerst een query in.")

    # Voorbeelden
    with st.expander("📚 Voorbeeld Queries"):
        st.code("""
-- Alle transacties van een specifiek aandeel
SELECT * FROM transactions WHERE ticker = 'AAPL';

-- Totaal aantal aandelen per ticker
SELECT ticker,
       SUM(CASE WHEN transaction_type = 'BUY' THEN quantity ELSE -quantity END) as total
FROM transactions
GROUP BY ticker;

-- Alle dividenden van dit jaar
SELECT * FROM dividends
WHERE strftime('%Y', ex_date) = '2024';

-- Transacties met hoogste kosten
SELECT * FROM transactions
ORDER BY fees DESC
LIMIT 10;
        """, language="sql")

# Tab 3: Schema Info
with tab3:
    st.subheader("Database Schema Informatie")

    schema_table = st.selectbox("Selecteer tabel voor schema:", tables, key="schema_select")

    if schema_table:
        st.write(f"### Schema van `{schema_table}`")

        schema = get_table_schema(schema_table)

        schema_df = pd.DataFrame(schema, columns=['cid', 'Naam', 'Type', 'Not Null', 'Default', 'Primary Key'])
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

# Tab 4: Database Stats
with tab4:
    st.subheader("Database Statistieken")

    # Algemene stats
    conn = get_connection()

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Aantal Records per Tabel")
        for table in tables:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            st.metric(table, count)

    with col2:
        st.write("#### Database Grootte")
        db_size = DB_PATH.stat().st_size / 1024  # KB
        st.metric("Database bestand", f"{db_size:.2f} KB")

        # Transactie stats
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE transaction_type = 'BUY'")
        buy_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE transaction_type = 'SELL'")
        sell_count = cursor.fetchone()[0]

        st.metric("BUY transacties", buy_count)
        st.metric("SELL transacties", sell_count)

    conn.close()

    # Recente activiteit
    st.divider()
    st.write("#### Laatste 10 Transacties")
    recent_df = get_table_data('transactions')
    if not recent_df.empty:
        recent_df_sorted = recent_df.sort_values('date', ascending=False).head(10)
        st.dataframe(recent_df_sorted, use_container_width=True, hide_index=True)
