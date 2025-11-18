"""
Tax Calculator voor Dividend Belastingen

Handles tax calculations for different jurisdictions and asset types:
- Belgian dividend tax (30%)
- Belgian REIT tax (48.9%)
- US withholding tax (15% with W-8BEN, 30% without)
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

DB_PATH = Path("data.db")


class TaxCalculator:
    """Calculator voor dividend belastingen."""

    def __init__(self):
        self._load_tax_rates()

    def _load_tax_rates(self):
        """Laad belastingtarieven uit de database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Haal tax settings op
        cursor.execute("SELECT setting_name, value FROM tax_settings")
        settings = {row[0]: float(row[1]) for row in cursor.fetchall()}

        # Default waarden als settings niet bestaan
        self.belgian_dividend_tax = settings.get('belgian_dividend_tax', 0.30)
        self.belgian_reit_tax = settings.get('belgian_reit_tax', 0.489)
        self.us_tax_with_w8 = settings.get('us_dividend_tax_with_w8', 0.15)
        self.us_tax_without_w8 = settings.get('us_dividend_tax_without_w8', 0.30)

        conn.close()

    def get_stock_info(self, ticker: str) -> Optional[Dict]:
        """Haal stock informatie op uit database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT asset_type, country, custom_dividend_tax_rate
            FROM stock_info
            WHERE ticker = ?
        """, (ticker,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'asset_type': result[0],
                'country': result[1],
                'custom_tax_rate': result[2]  # Can be None
            }
        return None

    def get_broker_w8_status(self, broker_name: str) -> bool:
        """Check of broker W-8BEN heeft."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT has_w8ben
            FROM broker_settings
            WHERE broker_name = ?
        """, (broker_name,))

        result = cursor.fetchone()
        conn.close()

        return bool(result[0]) if result else False

    def calculate_tax(
        self,
        bruto_amount: float,
        ticker: str,
        broker: Optional[str] = None,
        force_asset_type: Optional[str] = None,
        force_country: Optional[str] = None
    ) -> Dict:
        """
        Bereken belasting voor een dividend.

        Args:
            bruto_amount: Bruto dividend bedrag
            ticker: Stock ticker
            broker: Broker naam (optioneel, voor W-8BEN check)
            force_asset_type: Forceer een specifiek asset type (optioneel)
            force_country: Forceer een specifiek land (optioneel)

        Returns:
            Dict met:
                - belgian_tax: Belgische belasting
                - us_withholding: Amerikaanse bronheffing (indien van toepassing)
                - total_tax: Totale belasting
                - net_amount: Netto bedrag
                - tax_rate: Effectief belasting percentage
                - breakdown: Gedetailleerde breakdown
        """
        # Haal stock info op
        stock_info = self.get_stock_info(ticker)

        # Gebruik force parameters of database waarden
        asset_type = force_asset_type if force_asset_type else (
            stock_info['asset_type'] if stock_info else 'STOCK'
        )
        country = force_country if force_country else (
            stock_info['country'] if stock_info else None
        )
        custom_tax_rate = stock_info.get('custom_tax_rate') if stock_info else None

        # Initialize tax components
        belgian_tax = 0.0
        us_withholding = 0.0
        tax_breakdown = []

        # 1. Amerikaanse bronheffing (indien van toepassing)
        if country == "Verenigde Staten":
            has_w8 = self.get_broker_w8_status(broker) if broker else False
            us_rate = self.us_tax_with_w8 if has_w8 else self.us_tax_without_w8

            us_withholding = bruto_amount * us_rate
            w8_status = "met W-8BEN" if has_w8 else "zonder W-8BEN"
            tax_breakdown.append(f"🇺🇸 Amerikaanse bronheffing ({us_rate*100:.1f}% {w8_status}): €{us_withholding:.2f}")

        # 2. Belgische belasting
        # Check eerst voor custom tax rate
        if custom_tax_rate is not None:
            belgian_rate = custom_tax_rate
            belgian_tax = bruto_amount * belgian_rate
            tax_breakdown.append(f"🇧🇪 Belgische belasting ({belgian_rate*100:.1f}% - aangepast tarief): €{belgian_tax:.2f}")
        elif asset_type == "REIT":
            belgian_rate = self.belgian_reit_tax
            belgian_tax = bruto_amount * belgian_rate
            tax_breakdown.append(f"🇧🇪 Belgische REIT belasting ({belgian_rate*100:.1f}%): €{belgian_tax:.2f}")
        else:
            belgian_rate = self.belgian_dividend_tax
            belgian_tax = bruto_amount * belgian_rate
            tax_breakdown.append(f"🇧🇪 Belgische roerende voorheffing ({belgian_rate*100:.1f}%): €{belgian_tax:.2f}")

        # Totale belasting
        # Voor Amerikaanse aandelen: Amerikaanse bronheffing kan verrekend worden
        # maar voor deze calculatie tonen we het worst-case scenario
        total_tax = belgian_tax + us_withholding
        net_amount = bruto_amount - total_tax
        effective_rate = (total_tax / bruto_amount) if bruto_amount > 0 else 0

        # Voeg verrekening info toe voor US stocks
        if country == "Verenigde Staten" and us_withholding > 0:
            tax_breakdown.append(f"ℹ️ Amerikaanse bronheffing van €{us_withholding:.2f} kan verrekend worden in aangifte")

        return {
            'belgian_tax': belgian_tax,
            'us_withholding': us_withholding,
            'total_tax': total_tax,
            'net_amount': net_amount,
            'tax_rate': effective_rate,
            'breakdown': tax_breakdown,
            'asset_type': asset_type,
            'country': country or 'Onbekend'
        }

    def calculate_simple_tax(
        self,
        bruto_amount: float,
        ticker: str,
        broker: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Vereenvoudigde belasting berekening.

        Returns:
            Tuple van (tax_amount, net_amount)
        """
        result = self.calculate_tax(bruto_amount, ticker, broker)
        return result['total_tax'], result['net_amount']

    def get_tax_info_for_stock(self, ticker: str, broker: Optional[str] = None) -> str:
        """
        Haal leesbare belasting info op voor een stock.

        Returns:
            String met belasting informatie
        """
        stock_info = self.get_stock_info(ticker)

        if not stock_info:
            return "⚠️ Geen belasting info beschikbaar (configureer in Broker Settings)"

        asset_type = stock_info['asset_type']
        country = stock_info['country']
        custom_tax_rate = stock_info.get('custom_tax_rate')

        info_parts = []

        # Asset type info with custom tax rate if applicable
        if custom_tax_rate is not None:
            info_parts.append(f"📊 Type: {asset_type} ({custom_tax_rate*100:.1f}% aangepast tarief)")
        elif asset_type == "REIT":
            info_parts.append(f"📊 Type: REIT ({self.belgian_reit_tax*100:.1f}% BE belasting)")
        else:
            info_parts.append(f"📊 Type: {asset_type} ({self.belgian_dividend_tax*100:.1f}% BE belasting)")

        # Country info
        if country == "Verenigde Staten":
            if broker:
                has_w8 = self.get_broker_w8_status(broker)
                rate = self.us_tax_with_w8 if has_w8 else self.us_tax_without_w8
                w8_status = "✅ W-8BEN actief" if has_w8 else "❌ Geen W-8BEN"
                info_parts.append(f"🇺🇸 {rate*100:.1f}% US bronheffing ({w8_status})")
            else:
                info_parts.append(f"🇺🇸 15-30% US bronheffing (afhankelijk van W-8BEN)")
        elif country:
            info_parts.append(f"🌍 Land: {country}")

        return " | ".join(info_parts)


# Convenience functions voor gebruik in Streamlit
def calculate_dividend_tax(bruto_amount: float, ticker: str, broker: Optional[str] = None) -> Dict:
    """Bereken belasting voor een dividend."""
    calculator = TaxCalculator()
    return calculator.calculate_tax(bruto_amount, ticker, broker)


def get_tax_breakdown(bruto_amount: float, ticker: str, broker: Optional[str] = None) -> str:
    """Krijg een formatted string met tax breakdown."""
    result = calculate_dividend_tax(bruto_amount, ticker, broker)
    return "\n".join(result['breakdown'])


def get_net_dividend(bruto_amount: float, ticker: str, broker: Optional[str] = None) -> float:
    """Krijg netto dividend bedrag."""
    calculator = TaxCalculator()
    _, net = calculator.calculate_simple_tax(bruto_amount, ticker, broker)
    return net
