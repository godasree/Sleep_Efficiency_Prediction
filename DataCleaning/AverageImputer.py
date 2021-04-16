import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
from util_data import *
def averageImputer(sleep_activity_data_csv1):
    dataset_columns=list(sleep_activity_data_csv1.columns)
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='mean')
    imputer = imputer.fit(sleep_activity_data_csv1)
    sleep_activity_data_csv1= imputer.transform(sleep_activity_data_csv1)
    sleep_activity_data_csv1= pd.DataFrame(sleep_activity_data_csv1)
    sleep_activity_data_csv1.columns=dataset_columns

    # Imputer object using the mean strategy and
    # missing_values type for imputation
    return  sleep_activity_data_csv1






