import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def allocate_police(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allocate police officers based on burglary predictions.
    """
    scaler = MinMaxScaler(feature_range=(14, 100))
    df["officers"] = np.ceil(scaler.fit_transform(df[["prediction"]])).astype(int)

    return df