from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

def train(df):
    X = df[["Return","MA20","MA50","Momentum","Vol"]]
    y = df["Target"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rf = RandomForestClassifier(300)
    gb = GradientBoostingClassifier()

    rf.fit(Xs, y)
    gb.fit(Xs, y)

    prob = (rf.predict_proba(Xs)[:,1] + gb.predict_proba(Xs)[:,1]) / 2

    df["SignalProb"] = prob
    df["Signal"] = (prob > 0.55).astype(int)

    return df
