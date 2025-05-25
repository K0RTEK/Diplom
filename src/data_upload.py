import pandas as pd

def load_and_preprocess_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['receiveddatetime'])

    df['busnumber'] = df['busnumber'].fillna('UNKNOWN')
    for col in ['transactiontime', 'tarifficationdatetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    df['has_coords'] = ((df['lat'] != 0) | (df['lon'] != 0)).astype(int)
    df['carduid'] = df['carduid'].astype(str).str.strip()
    df['terminalid'] = df['terminalid'].astype(str).str.strip()

    return df
