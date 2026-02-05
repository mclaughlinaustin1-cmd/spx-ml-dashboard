def run(df):
    df["StrategyReturn"] = df["Return"] * df["Position"] * df["Size"]
    df["Equity"] = (1 + df["StrategyReturn"]).cumprod()

    peak = df["Equity"].cummax()
    drawdown = (df["Equity"] - peak) / peak

    stats = {
        "Total Return %": (df["Equity"].iloc[-1] - 1) * 100,
        "Win Rate %": (df["StrategyReturn"] > 0).mean() * 100,
        "Max Drawdown %": drawdown.min() * 100,
        "Sharpe (proxy)": df["StrategyReturn"].mean() / df["StrategyReturn"].std()
    }

    return df, stats
