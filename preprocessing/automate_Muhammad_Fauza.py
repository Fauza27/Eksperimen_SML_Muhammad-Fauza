import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    df = df.rename(columns={'V': 'Voltage', 'H': 'High', 'S': 'Soil type', 'M': 'Mine type'})

    NUMERICAL_FEATURES = ['Voltage', 'High']

    def cap_outliers_iqr(df, feature, multiplier=1.5):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
        return df

    for feature in NUMERICAL_FEATURES:
        df = cap_outliers_iqr(df, feature)

    scaler = StandardScaler()
    df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])

    return df

if __name__ == "__main__":
    df = pd.read_csv("landmine_raw.csv")

    df_processed = preprocess_data(df)

    df_processed.to_csv("preprocessing/landmine_preprocessing.csv", index=False)

    print("Preprocessing selesai. File disimpan di preprocessing/landmine_preprocessing.csv")

