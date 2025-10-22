import pandas as pd
import numpy as np

df = pd.read_csv("Bengaluru_House_Data.csv")

print("Column names:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isnull().sum())

df_dup.drop(columns=['availability'], inplace=True)

non_digit_indices = df[~df['total_sqft'].astype(str).str.strip().str.replace('.', '', 1).str.isdigit()].index.tolist()


def clean_sqft(x, idx):
    if pd.isna(x) or str(x).strip() == "":
        removed_indices.append(idx)
        return None

    x = str(x).strip()

    if '-' in x:
        parts = x.split('-')
        if len(parts) == 2:
            try:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                converted_indices.append(idx)
                return (low + high) / 2
            except:
                removed_indices.append(idx)
                return None

    try:
        return float(x)
    except:
        removed_indices.append(idx)
        return None


df_cleaned['size'] = df_cleaned['size'].astype(str)
df_cleaned['size_cleaned'] = df_cleaned['size'].str.extract(r'(\d+)')
df_cleaned['size_cleaned'] = pd.to_numeric(df_cleaned['size_cleaned'], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['size_cleaned'])
df_cleaned['size'] = df_cleaned['size_cleaned']
df_cleaned.drop(columns=['size_cleaned'], inplace=True)

df['balcony'] = df['balcony'].replace(r'^\s*$', 0.0, regex=True).astype(float)
df = df[~df['bath'].astype(str).str.strip().eq('')]

df.to_csv("Bengaluru_House_Data.csv", index=False)
