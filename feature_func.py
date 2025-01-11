# function used for projA2
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    temp = data[data[variable] > lower]
    temp_2 = temp[temp[variable] <= upper]
    return temp_2

def log_transform(data, col):
    temp = data.copy()
    temp[f'Log {col}'] = np.log(temp[col])
    return temp

def add_total_bedrooms(data):
    with_rooms = data.copy()
    temp = pd.to_numeric(data["Description"].str.extract(r'(\d+)\sof\swhich\sare\sbedrooms')[0])
    with_rooms["Bedrooms"] = temp
    return with_rooms

def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

def one_hot_encode(data, col):
    """
    Return the one-hot encoded DataFrame of our input data.
    
    Parameters
    -----------
    data: A DataFrame that may include non-numerical features.
    
    Returns
    -----------
    A one-hot encoded DataFrame that only contains numeric features.
    
    """
    ohe = OneHotEncoder()
    cat_features = col
    
    ohe.fit(data[cat_features])
    cat_data = ohe.transform(data[cat_features]).toarray()
    cat_df = pd.DataFrame(data = cat_data, columns = ohe.get_feature_names_out(), index = data.index)

    return data.join(cat_df).drop(columns = col), ohe.get_feature_names_out()
