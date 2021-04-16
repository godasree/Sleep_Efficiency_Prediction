import matplotlib

from DataCleaning.ImpGAIN import dataImputation_GAIN

matplotlib.use('Agg')
import pandas as pd
import numpy as np


def Convert_dict_window_to_window(users_Date, dict_users_dataset_window):
    user = list(set(users_Date["userId"]))
    window_dataset = []
    for i in range(len(user)):
        for j in range(dict_users_dataset_window[user[i]].shape[0]):
            window_dataset.append(dict_users_dataset_window[user[i]][j])
    window_dataset = np.array(window_dataset)
    return window_dataset


def Convert_Dict_to_window(users_Date, users_dataset_dictionary):
    user_Id = list(set(users_Date["userId"]))
    data_dictionary_window = {}
    number_of_windows = []
    for userId in user_Id:
        user_data_dict_1 = users_dataset_dictionary[userId]
        window = []
        for i in range(user_data_dict_1.shape[0] - 8 + 1):
            window.append([user_data_dict_1[i + j] for j in range(8)])
        window = np.array(window)
        number_of_windows.append(window.shape[0])
        data_dictionary_window[userId] = window
    return number_of_windows, data_dictionary_window


def Fillnan(behavioral_activity_sleep_data_csv):
    for i in range(behavioral_activity_sleep_data_csv.shape[0]):
        for j in range(behavioral_activity_sleep_data_csv.shape[1]):
            if behavioral_activity_sleep_data_csv.iloc[i, j] == -1:
                behavioral_activity_sleep_data_csv.iloc[i, j] = np.nan
    return behavioral_activity_sleep_data_csv


def Fetch_userdata_dictionary(dataset_user):
    users = list(set(dataset_user["userId"]))
    userdata_dictionary = {}
    for i in range(len(users)):
        user = dataset_user[dataset_user["userId"] == users[i]]
        user_data = np.array(user[user.columns[3:]])
        userdata_dictionary[users[i]] = user_data
    return userdata_dictionary


def get_mask_matrix(behavioral_activity_sleep_data_csv):
    users_Date = behavioral_activity_sleep_data_csv[list(behavioral_activity_sleep_data_csv.columns)[:3]]
    def masking(y):
        if y is True:
            y = 0
        else:
            y = 1
        return y
    maskmatrix= behavioral_activity_sleep_data_csv[list(behavioral_activity_sleep_data_csv.columns)[3:]].isnull().applymap(
        lambda y: masking(y))
    maskmatrix = pd.concat([users_Date, maskmatrix], axis=1)
    return maskmatrix


def get_mask_matrix_window(users_Date, maskmatrix):
    mask_data_dictionary = Fetch_userdata_dictionary(maskmatrix)
    mask_windownumber, mask_window_dict = Convert_Dict_to_window(users_Date, mask_data_dictionary)
    mask_matrix_window = Convert_dict_window_to_window(users_Date, mask_window_dict)
    return mask_matrix_window


def Delete_window(number_of_windows,Generator_probability_window,users,feature_name,max_value,min_value,users_Date):
    window_cummulative_sum = np.cumsum(number_of_windows)
    imputed_data_window_dictionary = {}
    for i in range(len(users)):
        if i == 0:
            imputed_data_window_dictionary[users[i]] = Generator_probability_window[:window_cummulative_sum[i]]
        else:
            imputed_data_window_dictionary[users[i]] = Generator_probability_window[window_cummulative_sum[i - 1]:window_cummulative_sum[i]]

    imputed_data_dictionary = {}
    for i in range(len(users)):
        user_Id_window = imputed_data_window_dictionary[users[i]]
        data_users = []
        for j in range(user_Id_window.shape[0]):
            if j != user_Id_window.shape[0] - 1:
                data_users.append(user_Id_window[j, 0])
            else:
                for k in range(user_Id_window[j].shape[0]):
                    data_users.append(user_Id_window[j][k])
        imputed_data_dictionary[users[i]] = np.array(data_users)
    impute_data_gain = []
    for i in range(len(users)):
        for j in range(imputed_data_dictionary[users[i]].shape[0]):
            impute_data_gain.append(imputed_data_dictionary[users[i]][j])
    impute_data_gain = np.array(impute_data_gain)
    impute_data_frame= pd.DataFrame(np.array(impute_data_gain), columns=feature_name)
    impute_gain_data_standard = pd.concat([users_Date, impute_data_frame], axis=1)  # %
    impute_data_gain = min_value + (max_value - min_value) * impute_data_frame
    impute_data_gain = pd.concat([users_Date, impute_data_gain], axis=1)
    return impute_gain_data_standard, impute_data_gain


if __name__ == '__main__':
    dataset_path = 'C:/DBProject/Sleep_Efficiency_Prediction/datasets/full_data_sleeps.csv'
    imputeddata_path = 'C:/DBProject/Sleep_Efficiency_Prediction/datasets/imputed_data_GAIN.csv'
    epouh = 19000
    ##loading excel datasets into pandas dataframe
    behavioral_activity_sleep_data_csv = pd.read_csv(dataset_path)
    #filling missing data with Nan Value
    behavioral_activity_sleep_data_csv = Fillnan(behavioral_activity_sleep_data_csv)
    feature_name = list(behavioral_activity_sleep_data_csv.columns)[3:]
    behavioral_activity_sleep_data = behavioral_activity_sleep_data_csv.fillna(0.8)
    max_value = behavioral_activity_sleep_data_csv[behavioral_activity_sleep_data_csv.columns[3:]].max()
    min_value = behavioral_activity_sleep_data_csv[behavioral_activity_sleep_data_csv.columns[3:]].min()
    users_Date = behavioral_activity_sleep_data_csv[['userId', 'month', 'date']]
    users = list(set(behavioral_activity_sleep_data_csv["userId"]))
    behavioral_activity_sleep_data_standard = (behavioral_activity_sleep_data[behavioral_activity_sleep_data.columns[3:]]-min_value)/(max_value-min_value)
    behavioral_activity_sleep_data_standard =  pd.concat([users_Date,behavioral_activity_sleep_data_standard],axis=1)
    users_dataset_dictionary = Fetch_userdata_dictionary(behavioral_activity_sleep_data_standard)
    number_of_windows, dict_users_dataset_window = Convert_Dict_to_window(users_Date,users_dataset_dictionary)
    window_dataset = Convert_dict_window_to_window(users_Date,dict_users_dataset_window)
    maskmatrix= get_mask_matrix(behavioral_activity_sleep_data_csv)
    mask_matrix_window = get_mask_matrix_window(users_Date,maskmatrix)
    Generator_probability_window = dataImputation_GAIN(mask_matrix_window,window_dataset,epouh).reshape((-1,8,len(feature_name)))
    imputed_data_minmax_norm,imputed_data = Delete_window(number_of_windows,Generator_probability_window,users,feature_name,max_value,min_value,users_Date)
    imputed_data.to_csv(imputeddata_path,index=0)
