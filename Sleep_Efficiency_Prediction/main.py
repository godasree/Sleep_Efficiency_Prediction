import csv
import argparse
from LSTMphase1_model import *
from LSTMphase2_model import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def datacleaning_average():
    userId = -1
    imputed_data_average = []
    _, userId_date, data_sleep_activity = fetch_dataset("FULLDATA")
    for datarow in data_sleep_activity:
        if userId != int(datarow[0]):
            userId = int(datarow[0])
            vartemparray = []
        datarow = np.array(datarow).reshape(-1, len(datarow))[:, 1:]
        if len(vartemparray) == 0:
            columnmean = np.zeros([1, 14])
        else:
            columnmean = np.mean(np.array(vartemparray), axis = 0)
        if np.any(datarow[0, :] == -1, axis = 0):
            minus_index = np.argwhere(datarow == -1)
            datarow[0, minus_index] = columnmean[0, minus_index]
        vartemparray.append(datarow)
        imputed_data_average.append(datarow)
    imputed_data_average = np.concatenate([userId_date, np.array(imputed_data_average).reshape(-1, 14)], axis = 1)
    np.savetxt("C:/DBProject/Sleep_Efficiency_Prediction/datasets/imputed_data_AVERAGE.csv", imputed_data_average, header=",".join(get_feature_names()), delimiter=",", comments='', fmt='%s')
    return imputed_data_average

def datacleaning_BLANK():
    imputed_data_blank = []
    zeromatrix = np.zeros([1, 14])
    _, userId_date, data_sleep_activity =fetch_dataset("FULLDATA")
    for datarow in data_sleep_activity[:, 1:]:
        datarow = np.array(datarow).reshape(-1, len(datarow))
        if np.any(datarow[0, :] == -1, axis=0):
            missingdataindex= np.argwhere(datarow == -1)
            datarow[0, missingdataindex] = zeromatrix[0, missingdataindex]
        imputed_data_blank.append(datarow)
    imputed_data_blank = np.concatenate([userId_date, np.array(imputed_data_blank).reshape(-1, 14)], axis=1)
    np.savetxt("C:/DBProject/Sleep_Efficiency_Prediction/datasets/imputed_data_BLANK.csv", imputed_data_blank, header=",".join(get_feature_names()), delimiter=",",
                   comments='', fmt='%s')
    return imputed_data_blank

def datacleaning_GAIN():
    cleaneddataset, _, _ = fetch_dataset("gain")
    return cleaneddataset

def fetch_dataset(impute_method):
    if impute_method == "blank":
        dataset = np.genfromtxt("C:/DBProject/Sleep_Efficiency_Prediction/datasets/imputed_data_BLANK.csv", delimiter=',')[1:]
    elif impute_method == "gain":
        dataset = np.genfromtxt("C:/DBProject/Sleep_Efficiency_Prediction/datasets/imputed_data_GAIN.csv", delimiter=',')[1:]
    elif impute_method == "average ":
        dataset = np.genfromtxt("C:/DBProject/Sleep_Efficiency_Prediction/datasets/imputed_data_AVERAGE.csv", delimiter=',')[1:]
    else:
        dataset = np.genfromtxt("C:/DBProject/Sleep_Efficiency_Prediction/datasets/full_data_sleeps.csv", delimiter=',')[1:]
    userId_Date = dataset[:, 0:3]
    data_sleep_activity= np.concatenate([userId_Date[:, 0:1], dataset[:, 3:17]], axis = 1)
    return dataset, userId_Date, data_sleep_activity

def get_feature_names():
    data_sleep_activity = np.genfromtxt("C:/DBProject/Sleep_Efficiency_Prediction/datasets/full_data_sleeps.cs.csv", delimiter=',', names = True)[1:]
    return list(data_sleep_activity.dtype.names)


def getwindowlist(users, data_sleep_activity, column_size = 8):
    data_sleep_activity = np.concatenate([users, data_sleep_activity], axis = 1)
    userwindow = []
    user_window_sliding = []
    for userid in np.unique(users):
        user_data = data_sleep_activity[np.where(data_sleep_activity[:, 0] == userid)][:, 1:]
        userwindow.append(user_data)
    for window in userwindow:
        i = 0
        while i + column_size <= window.shape[0]:
            user_window_sliding.append(window[i:i+column_size, :])
            i = i + 1
    return user_window_sliding

def report_learning_loss(modelname, lossarray):
    with open('result/loss_result_{}.csv'.format(modelname), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Phase1_step1", "Phase1_step2", "Phase1_step3", "Phase1_step4", "Phase1_step5", "Phase1_step6",
                     "Phase1_step7", "Phase2"])
        writer.writerow(lossarray)

def gettrain_testdata(cleaned_data_set):
    trainwindow = []
    testwindow = []
    userwindowlist = getwindowlist(cleaned_data_set[:, 0:1], cleaned_data_set[:, 3:])
    userwindowlist.reverse()
    user = 0
    testdatacount = 0
    for userwindow in userwindowlist:
        if testdatacount < 7 and userwindow[-1, 3] != -1:
            testdatacount += 1
            testwindow.append(user)
        else:
            trainwindow.append(user)
        user += 1
        if user % 35 == 0:
            testdatacount = 0
    return trainwindow, testwindow

def get_metadata(cleaneddata, lossarray, trainwindow, testwindow):
    flipped_cleandata= np.flip(cleaneddata.copy(), 0)
    metadataset = np.genfromtxt("C:/DBProject/Sleep_Efficiency_Prediction/datasets/full_meta-data_sleeps.csv", delimiter=',')[1:, 1:]
    users = metadataset.shape[0]
    minmaxscaler = MinMaxScaler()
    loss_array_scaled = minmaxscaler.fit_transform(np.array(lossarray).reshape(-1, 1)).T
    loss_array_tile = np.tile(loss_array_scaled, [users * 35, 1])
    meta_array_scaled = minmaxscaler.fit_transform(metadataset)
    metadataoutput=[]
    for i in range(users):
        for j in range(35):
            metadataoutput.append(meta_array_scaled[i, :])

    metadataoutput = np.flip(np.concatenate([np.array(metadataoutput), loss_array_tile, flipped_cleandata[:, :7, :].reshape(-1, 98)], axis = 1), 0)
    trainmetadata = metadataoutput[trainwindow]
    testmetadata = metadataoutput[testwindow]
    return trainmetadata, testmetadata

def preprocesstrainTestData(cleaneddata, trainwindow, testwindow):
    data_sleep_activity = cleaneddata[:, 3:17]
    minval = np.min(data_sleep_activity, axis = 0)[None, :]
    diff = data_sleep_activity.max(axis = 0) - data_sleep_activity.min(axis = 0)[None, :]
    sleep_efficiency = data_sleep_activity[:, 3]
    preprocessed_data_sleep_activity = (data_sleep_activity - minval) / diff
    preprocessed_data_sleep_activity[:, 3] = sleep_efficiency
    userswindowlist = np.array(getwindowlist(cleaneddata[:, 0:1], preprocessed_data_sleep_activity)).tolist()
    userswindowlist.reverse()
    trainingdata = np.array(userswindowlist)[trainwindow]
    testingdata = np.array(userswindowlist)[testwindow]
    return userswindowlist, trainingdata, testingdata

def report_learning_alpha(modelname, alphas, stepsize):
    with open('result/attention_output_{}.csv'.format(modelname), 'w') as f:
        writer = csv.writer(f)
        user= 1
        numberofday = 1
        writer.writerow(
            ["UserId", "Day", "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6", "Step 7"][:stepsize + 2])
        for row in alphas:
            list_to_write = [user, numberofday]
            if isinstance(row, np.float32):
                list_to_write.append(row)
            else:
                list_to_write.extend(row)
            writer.writerow(list_to_write)
            if numberofday == 7:
                numberofday = 1
                user += 1
            else:
                numberofday += 1



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--modelname', type=str, dest='modelname_output', help='model Name to be printed in output')
    argparser.add_argument('--dataimpute', type=str, dest='cleaningMethod',
                        help='Method for cleaning the data GAIN, AVERAGE, and BLANK')
    arguments = argparser.parse_args()
    modelname = arguments.modelname_output
    cleaning_method = arguments.cleaningMethod
    if cleaning_method == "gain":
        datacleaning_GAIN()
    elif cleaning_method == "blank":
        datacleaning_BLANK()
    else:
        datacleaning_average()
    lstmphase1_parameter ={"epoch" : 350,"batchsize" : 30,"learningrate" : 1e-4,"step1hiddensize": 200,"step2hiddensize": 75,"step3hiddensize": 50,"step4hiddensize": 100,"step5hiddensize": 90,"step6hiddensize": 100, "step7hiddensize": 300,"keepprobability" : 0.7}
    lstmphase2_parameter = {"epoch" : 350,"batchsize" : 30, "learningrate" : 1e-4,"querysize" : 70, "metadatahiddensize1": 90,"metadatahiddensize2": 50,"keepprobability" : 0.7}
    cleaneddata, _, _ = fetch_dataset(cleaning_method)
    trainwindow, testwindow = gettrain_testdata(cleaneddata)
    userswindowlist, trainingdata, testingdata = preprocesstrainTestData(cleaneddata, trainwindow, testwindow)
    lossarray = []
    training_phase2_hiddenlist = []
    testing_phase2_hiddenlist = []
    block_hidden_list = []
    for step_size in range(1, 8):
        name = "Phase1_{}_{}".format(modelname, step_size)
        loss, alphas, y_hat, phase2_training_hiddenvector, phase2_testing_hiddenvector = phase1train(name, trainingdata, testingdata, step_size,
                                                                                   lstmphase1_parameter['batchsize'],
                                                                                   lstmphase1_parameter['learningrate'],
                                                                                   lstmphase1_parameter[
                                                                                       'step{}hiddensize'.format(
                                                                                           step_size)],
                                                                                   lstmphase1_parameter['epoch']
                                                                                   )
        lossarray.append(loss)
        block_hidden_list.append(lstmphase1_parameter['step{}hiddensize'.format(step_size)])
        training_phase2_hiddenlist.append(phase2_training_hiddenvector)
        testing_phase2_hiddenlist.append(phase2_testing_hiddenvector)

        if step_size > 1:
            report_learning_alpha(name, alphas, step_size)
    trainmetadata, testmetadata = get_metadata(np.array(userswindowlist), lossarray, trainwindow,
                                                 testwindow)
    block_hidden_list.append(lstmphase2_parameter['querysize'])
    name = "Phase2_{}".format(modelname)
    loss, alphas, y_hat = phase2_trainorload(name,
                                             training_phase2_hiddenlist, trainmetadata, trainingdata[:, 7, 3],
                                             testing_phase2_hiddenlist, testmetadata, testingdata[:, 7, 3],
                                             lstmphase2_parameter['batchsize'],
                                             lstmphase2_parameter['learningrate'],
                                             block_hidden_list,
                                             108,
                                            [ lstmphase2_parameter['metadatahiddensize1'],
                                              lstmphase2_parameter['metadatahiddensize2']],
                                             lstmphase2_parameter['epoch'],
                                             lstmphase2_parameter['keepprobability'],
                                             )

    lossarray.append(loss)
    report_learning_alpha(name, alphas, 7)
    report_learning_loss(modelname, lossarray)


