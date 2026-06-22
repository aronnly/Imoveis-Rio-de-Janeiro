import argparse
import time
from pathlib import Path
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ITBI_mapa_excel_valores_corrigidos_IPCA_mar_2026.xlsx"
OUT_PATH = BASE_DIR / "data" / "geocoded_logradouros.csv"
SHEET_NAME = "Base_Normalizada"


def load_addresses():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl", usecols=["bairro", "logradouro", "endereco_mapa"])
    for col in ["bairro", "logradouro", "endereco_mapa"]:
        df[col] = df[col].astype("string").str.strip()
    df = df.dropna(subset=["bairro", "logradouro", "endereco_mapa"])
    return df.drop_duplicates(["bairro", "logradouro"]).sort_values(["bairro", "logradouro"])


def load_existing():
    if not OUT_PATH.exists():
        return pd.DataFrame(columns=["bairro", "logradouro", "endereco_mapa", "latitude", "longitude", "status_geocodificacao"])
    existing = pd.read_csv(OUT_PATH)
    for col in ["bairro", "logradouro", "endereco_mapa", "status_geocodificacao"]:
        if col in existing.columns:
            existing[col] = existing[col].astype("string").str.strip()
    return existing


def geocode_one(geolocator, address, retries=3):
    for _ in range(retries):
        try:
            return geolocator.geocode(address, timeout=20, exactly_one=True, country_codes="br")
        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError):
            time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.1)
    args = parser.parse_args()
    addresses = load_addresses()
    existing = load_existing()
    done_keys = set(zip(existing["bairro"].astype(str), existing["logradouro"].astype(str))) if not existing.empty else set()
    pending = addresses[~addresses.apply(lambda r: (str(r["bairro"]), str(r["logradouro"])) in done_keys, axis=1)].copy()
    if args.limit and args.limit > 0:
        pending = pending.head(args.limit)
    geolocator = Nominatim(user_agent="itbi_rio_dashboard_local")
    rows = []
    for i, row in pending.reset_index(drop=True).iterrows():
        address = row["endereco_mapa"]
        print(f"{i + 1}/{len(pending)} — {address}")
        loc = geocode_one(geolocator, address)
        if loc:
            rows.append({"bairro": row["bairro"], "logradouro": row["logradouro"], "endereco_mapa": address, "latitude": loc.latitude, "longitude": loc.longitude, "status_geocodificacao": "ok"})
        else:
            rows.append({"bairro": row["bairro"], "logradouro": row["logradouro"], "endereco_mapa": address, "latitude": None, "longitude": None, "status_geocodificacao": "não encontrado"})
        partial = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        partial.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
        time.sleep(args.sleep)
    final = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["bairro", "logradouro"], keep="last")
    final.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Arquivo salvo em: {OUT_PATH}")


if __name__ == "__main__":
    main()
