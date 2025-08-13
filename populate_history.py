# populate_history.py
#
# DESCRIZIONE:
# Script una-tantum per popolare il file dati storico con un lookback
# di 10 anni. Da eseguire manualmente una sola volta.

import sys
import os
from datetime import datetime, timedelta

# Aggiungi la cartella 'src' al path per trovare i nostri moduli
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from data_fetcher import DataFetcher

def populate():
    """Scarica e salva 10 anni di dati storici."""
    print("="*50)
    print("INIZIO POPOLAMENTO STORICO DATI")
    print("="*50)

    try:
        # 1. Definisci il lookback desiderato
        lookback_years = 10
        ticker = Config.TICKER
        print(f"Ticker: {ticker}")
        print(f"Lookback desiderato: {lookback_years} anni")

        # 2. Calcola le date
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_years * 365)).strftime('%Y-%m-%d')

        # 3. Scarica i dati usando il metodo corretto
        fetcher = DataFetcher()
        df_historical = fetcher.fetch_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )

        if df_historical is not None and not df_historical.empty:
            # 4. Salva i dati usando il metodo corretto
            # Assumiamo che Config.HISTORICAL_DATA_FILE sia definito in config.py
            # e punti a 'data/historical_data.csv'
            saved_path = fetcher.save_data(df_historical, filename=Config.HISTORICAL_DATA_FILE)
            print(f"\n✅ Dati salvati con successo in: {saved_path}")
            print(f"Totale record scaricati: {len(df_historical)}")
        else:
            print("\n❌ Errore: Nessun dato scaricato.")

    except Exception as e:
        print(f"\nERRORE CRITICO DURANTE IL POPOLAMENTO: {e}")

    print("\n" + "="*50)
    print("POPOLAMENTO COMPLETATO")
    print("="*50)


if __name__ == "__main__":
    populate()
