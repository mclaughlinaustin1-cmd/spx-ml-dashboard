def apply(df):
    df["Position"] = df["Signal"].shift()
    return df.dropna()

