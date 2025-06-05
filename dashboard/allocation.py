def allocate_police(df_predictions):
    """
    Allocate police officers based on the risk factor.
    """
    max_pred = max(df_predictions['prediction'])
    df_predictions['risk'] = round((df_predictions['prediction'] / max_pred), 3) # Calculate risk factor
    df_predictions['officers'] = round(df_predictions['risk'] * 100)

    return df_predictions
