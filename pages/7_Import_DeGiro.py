"""
DeGiro CSV Import
Import dividend data from DeGiro CSV exports
"""

import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tax_calculator import TaxCalculator

DB_PATH = Path("data.db")


def parse_degiro_csv(uploaded_file):
    """
    Parse DeGiro CSV export and extract dividend information.

    Returns:
        DataFrame with grouped dividend data per ISIN
    """
    try:
        # Read CSV - don't use decimal/thousands parsing yet
        df = pd.read_csv(uploaded_file, encoding='utf-8')

        # Filter for dividend-related transactions
        dividend_mask = df['Omschrijving'].str.contains('Dividend', na=False, case=False)
        dividend_df = df[dividend_mask].copy()

        if dividend_df.empty:
            return None, "Geen dividend transacties gevonden in CSV"

        # Convert amount column from European format to float
        # The amount is in Unnamed: 8, not in Mutatie (which contains currency)
        def parse_amount(value):
            if pd.isna(value) or value == '':
                return 0.0
            # Remove dots (thousands separator) and replace comma with dot
            value_str = str(value).replace('.', '').replace(',', '.')
            try:
                return float(value_str)
            except ValueError:
                return 0.0

        # Find the unnamed column after Mutatie that contains the actual amounts
        amount_col = 'Unnamed: 8' if 'Unnamed: 8' in df.columns else df.columns[df.columns.get_loc('Mutatie') + 1]
        dividend_df['Mutatie_parsed'] = dividend_df[amount_col].apply(parse_amount)

        # Group by ISIN and process
        grouped_results = []

        for isin, group in dividend_df.groupby('ISIN'):
            # Separate dividend receipts and tax payments
            dividends = group[~group['Omschrijving'].str.contains('belasting', case=False, na=False)]
            taxes = group[group['Omschrijving'].str.contains('belasting', case=False, na=False)]

            if dividends.empty:
                continue

            # Get basic info from first dividend entry
            first_entry = dividends.iloc[0]
            product_name = first_entry['Product']

            # Calculate totals (using parsed amounts)
            bruto_received = dividends['Mutatie_parsed'].sum()
            tax_withheld = abs(taxes['Mutatie_parsed'].sum()) if not taxes.empty else 0.0

            # Calculate bruto before tax
            bruto_amount = bruto_received + tax_withheld

            # Calculate effective tax rate
            effective_tax_rate = (tax_withheld / bruto_amount * 100) if bruto_amount > 0 else 0

            # Get dates
            dividend_dates = dividends['Datum'].tolist()
            latest_date = dividends['Datum'].iloc[0] if not dividends.empty else None

            # Get currency from FX column
            currency = first_entry['FX'] if pd.notna(first_entry['FX']) and first_entry['FX'] != '' else 'EUR'

            grouped_results.append({
                'Product': product_name,
                'ISIN': isin,
                'Datum': latest_date,
                'Currency': currency,
                'Bruto': round(bruto_amount, 2),
                'Belasting': round(tax_withheld, 2),
                'Netto': round(bruto_received, 2),
                'Tax %': round(effective_tax_rate, 1),
                'Aantal transacties': len(group)
            })

        result_df = pd.DataFrame(grouped_results)
        return result_df, None

    except Exception as e:
        import traceback
        return None, f"Fout bij parsen CSV: {str(e)}\n{traceback.format_exc()}"


def get_ticker_from_isin(isin):
    """Lookup ticker from ISIN in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT ticker FROM transactions WHERE isin = ? LIMIT 1", (isin,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None


def analyze_tax_rates(df):
    """
    Analyze tax rates and add warnings for unexpected values.

    Returns:
        DataFrame with added 'Warning' column and ticker information
    """
    tax_calc = TaxCalculator()
    warnings = []
    expected_rates = []
    asset_types = []
    tickers = []

    for _, row in df.iterrows():
        isin = row['ISIN']
        actual_rate = row['Tax %']

        # Ensure actual_rate is numeric
        try:
            actual_rate = float(actual_rate)
        except (ValueError, TypeError):
            actual_rate = 0.0

        # Try to get ticker and tax info
        ticker = get_ticker_from_isin(isin)
        tickers.append(ticker if ticker else "-")

        if ticker:
            stock_info = tax_calc.get_stock_info(ticker)

            if stock_info:
                asset_type = stock_info['asset_type']
                country = stock_info['country']
                asset_types.append(asset_type)

                # Determine expected rate
                if asset_type == 'REIT':
                    expected_rate = tax_calc.belgian_reit_tax * 100
                    expected_rates.append(f"{expected_rate:.1f}% (REIT)")
                elif country == 'Verenigde Staten':
                    expected_rates.append("15-45% (US)")
                else:
                    expected_rate = tax_calc.belgian_dividend_tax * 100
                    expected_rates.append(f"{expected_rate:.1f}%")

                # Check for warnings
                tolerance = 2.0  # 2% tolerance
                if asset_type == 'REIT':
                    if abs(actual_rate - expected_rate) > tolerance:
                        warnings.append(f"⚠️ Verwacht ~{expected_rate:.1f}% (REIT)")
                    else:
                        warnings.append("✅")
                elif country == 'Verenigde Staten':
                    # US stocks can vary (15% + 30% BE = 41-45% depending on calculation)
                    if actual_rate < 10 or actual_rate > 50:
                        warnings.append(f"⚠️ Onverwacht voor US aandeel")
                    else:
                        warnings.append("✅")
                else:
                    if abs(actual_rate - expected_rate) > tolerance:
                        warnings.append(f"⚠️ Verwacht ~{expected_rate:.1f}%")
                    else:
                        warnings.append("✅")
            else:
                warnings.append("ℹ️ Configureer asset type in Broker Settings")
                expected_rates.append("-")
                asset_types.append("-")
        else:
            warnings.append("ℹ️ ISIN niet gevonden in portfolio")
            expected_rates.append("-")
            asset_types.append("-")

    df['Ticker'] = tickers
    df['Type'] = asset_types
    df['Verwacht'] = expected_rates
    df['Status'] = warnings

    return df


def import_dividends_to_db(df, received=True, tax_paid=True):
    """
    Import dividends from DataFrame to database.

    Returns:
        Tuple of (success_count, error_list)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    success_count = 0
    errors = []

    for idx, row in df.iterrows():
        try:
            ticker = row.get('Ticker', '-')
            isin = row['ISIN']

            # Skip if no ticker found
            if ticker == '-' or not ticker:
                errors.append(f"Regel {idx+1}: ISIN {isin} niet gekoppeld aan ticker")
                continue

            # Parse date (format: DD-MM-YYYY from DeGiro)
            date_str = row['Datum']
            try:
                date_obj = datetime.strptime(date_str, '%d-%m-%Y')
            except ValueError:
                # Try alternative format
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')

            # Insert dividend
            cursor.execute("""
                INSERT INTO dividends (
                    ticker, isin, ex_date, bruto_amount, currency,
                    received, tax_paid, withheld_amount, net_received, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                isin,
                date_obj.strftime('%Y-%m-%d'),
                float(row['Bruto']),
                row['Currency'],
                1 if received else 0,
                1 if tax_paid else 0,
                float(row['Belasting']),
                float(row['Netto']),
                f"Geïmporteerd van DeGiro CSV op {datetime.now().strftime('%Y-%m-%d')}"
            ))

            success_count += 1

        except Exception as e:
            errors.append(f"Regel {idx+1} ({row.get('Product', 'Unknown')}): {str(e)}")

    conn.commit()
    conn.close()

    return success_count, errors


# --------- Streamlit UI ---------
st.set_page_config(page_title="Import DeGiro", page_icon="📥")
st.title("📥 Import DeGiro CSV")

st.markdown("""
Upload je DeGiro account overzicht (CSV) om dividend transacties te importeren.
Het systeem detecteert automatisch dividenden en bijbehorende belastingen.
""")

# File uploader
uploaded_file = st.file_uploader(
    "Selecteer DeGiro CSV bestand",
    type=['csv'],
    help="Download je account overzicht via DeGiro > Account > Transacties exporteren"
)

if uploaded_file is not None:
    st.info(f"📄 Bestand geladen: {uploaded_file.name}")

    with st.spinner("CSV aan het parsen..."):
        parsed_df, error = parse_degiro_csv(uploaded_file)

    if error:
        st.error(error)
    elif parsed_df is not None and not parsed_df.empty:
        st.success(f"✅ {len(parsed_df)} dividend transacties gevonden")

        # Analyze tax rates
        with st.spinner("Belastingen aan het analyseren..."):
            analyzed_df = analyze_tax_rates(parsed_df)

        st.subheader("📊 Preview van dividend data")

        # Display with color coding for warnings
        st.dataframe(
            analyzed_df,
            use_container_width=True,
            column_config={
                "Datum": st.column_config.TextColumn("Datum"),
                "Product": st.column_config.TextColumn("Aandeel", width="medium"),
                "ISIN": st.column_config.TextColumn("ISIN", width="small"),
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Bruto": st.column_config.NumberColumn("Bruto €", format="€%.2f"),
                "Belasting": st.column_config.NumberColumn("Belasting €", format="€%.2f"),
                "Netto": st.column_config.NumberColumn("Netto €", format="€%.2f"),
                "Tax %": st.column_config.NumberColumn("Tax %", format="%.1f%%"),
                "Verwacht": st.column_config.TextColumn("Verwacht", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small"),
            }
        )

        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Totaal Bruto", f"€{analyzed_df['Bruto'].sum():.2f}")
        with col2:
            st.metric("Totaal Belasting", f"€{analyzed_df['Belasting'].sum():.2f}")
        with col3:
            st.metric("Totaal Netto", f"€{analyzed_df['Netto'].sum():.2f}")
        with col4:
            avg_tax = (analyzed_df['Belasting'].sum() / analyzed_df['Bruto'].sum() * 100)
            st.metric("Gem. Tax %", f"{avg_tax:.1f}%")

        # Warnings section
        warnings_count = analyzed_df['Status'].str.contains('⚠️', na=False).sum()
        unknown_count = analyzed_df['Status'].str.contains('ℹ️', na=False).sum()

        if warnings_count > 0:
            st.warning(f"⚠️ {warnings_count} aandelen met afwijkende belastingpercentages gedetecteerd")

        if unknown_count > 0:
            st.info(f"ℹ️ {unknown_count} aandelen moeten nog geconfigureerd worden in Broker Settings")

        st.divider()

        # Import options
        st.subheader("⚙️ Import opties")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            default_received = st.checkbox(
                "Markeer als ontvangen",
                value=True,
                help="Vink aan als je deze dividenden al ontvangen hebt"
            )

        with col_opt2:
            default_tax_paid = st.checkbox(
                "Markeer belasting als betaald",
                value=True,
                help="Vink aan als de belasting al ingehouden is"
            )

        # Import button
        if st.button("💾 Importeer alle dividenden", type="primary"):
            with st.spinner("Dividenden aan het importeren..."):
                success_count, errors = import_dividends_to_db(
                    analyzed_df,
                    received=default_received,
                    tax_paid=default_tax_paid
                )

            if success_count > 0:
                st.success(f"✅ {success_count} dividenden succesvol geïmporteerd!")

            if errors:
                st.error(f"❌ {len(errors)} fouten bij importeren:")
                for error in errors:
                    st.write(f"- {error}")

            if success_count > 0:
                st.info("💡 Ga naar de Dividend Kalender pagina om je geïmporteerde dividenden te bekijken")
    else:
        st.warning("Geen dividend transacties gevonden in het CSV bestand")

else:
    # Instructions when no file uploaded
    st.info("👆 Upload een CSV bestand om te beginnen")

    with st.expander("📖 Hoe werkt het?"):
        st.markdown("""
        ### DeGiro CSV exporteren:
        1. Log in op DeGiro
        2. Ga naar **Account** > **Transacties**
        3. Selecteer de gewenste periode
        4. Klik op **Exporteren** en kies CSV

        ### Wat wordt geïmporteerd?
        - Dividend ontvangsten
        - Dividendbelasting (automatisch gekoppeld)
        - Bruto, netto en belasting bedragen
        - Effectieve belastingpercentages

        ### Validaties:
        - ✅ Verwachte belastingpercentages (30%, 48.9% voor REITs)
        - ⚠️ Waarschuwingen bij afwijkende percentages
        - ℹ️ Detectie van nog te configureren aandelen
        """)
