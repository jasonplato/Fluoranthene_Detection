import pandas as pd
import numpy as np


def get_data_set(flag):
    """
    :param flag - set 0 for training dataset, set 1 for testing dataset
    :return: features - a matrix with 29 columns feature-vectors
    :return: labels - a corresponding vector indicating ground truth label
    """
    if flag == 0:
        df = pd.read_csv('Otonabee_Fluoranthene_training.csv', header=None)
    else:
        df = pd.read_csv('Otonabee_Fluoranthene_test.csv', header=None)
    df['1_mean'] = df.iloc[:, 1: 112].mean(axis=1)
    df['2_mean'] = df.iloc[:, 112:223].mean(axis=1)
    df['3_mean'] = df.iloc[:, 223:334].mean(axis=1)
    df['4_mean'] = df.iloc[:, 334:445].mean(axis=1)
    df['5_mean'] = df.iloc[:, 445:556].mean(axis=1)
    df['6_mean'] = df.iloc[:, 556:667].mean(axis=1)
    df['7_mean'] = df.iloc[:, 667:778].mean(axis=1)
    df['8_mean'] = df.iloc[:, 778:889].mean(axis=1)
    df['9_mean'] = df.iloc[:, 889:1000].mean(axis=1)
    df['10_mean'] = df.iloc[:, 1000:1111].mean(axis=1)
    df['11_mean'] = df.iloc[:, 1111:1222].mean(axis=1)
    df['12_mean'] = df.iloc[:, 1222:1333].mean(axis=1)
    df['13_mean'] = df.iloc[:, 1333:1444].mean(axis=1)
    df['14_mean'] = df.iloc[:, 1444:1555].mean(axis=1)
    df['15_mean'] = df.iloc[:, 1555:1666].mean(axis=1)
    df['16_mean'] = df.iloc[:, 1666:1777].mean(axis=1)
    df['17_mean'] = df.iloc[:, 1777:1888].mean(axis=1)
    df['18_mean'] = df.iloc[:, 1888:1999].mean(axis=1)
    df['19_mean'] = df.iloc[:, 1999:2110].mean(axis=1)
    df['20_mean'] = df.iloc[:, 2110:2221].mean(axis=1)
    df['21_mean'] = df.iloc[:, 2221:2332].mean(axis=1)
    df['22_mean'] = df.iloc[:, 2332:2443].mean(axis=1)
    df['23_mean'] = df.iloc[:, 2443:2554].mean(axis=1)
    df['24_mean'] = df.iloc[:, 2554:2665].mean(axis=1)
    df['25_mean'] = df.iloc[:, 2665:2776].mean(axis=1)
    df['26_mean'] = df.iloc[:, 2776:2887].mean(axis=1)
    df['27_mean'] = df.iloc[:, 2887:2998].mean(axis=1)
    df['28_mean'] = df.iloc[:, 2998:3109].mean(axis=1)
    df['29_mean'] = df.iloc[:, 3109:3220].mean(axis=1)
    df['label'] = pd.cut(df.iloc[:, 0], [-np.inf, 0.5, np.inf], labels=[-1,1], right=False)


    features = df[
        ['1_mean', '2_mean', '3_mean', '4_mean', '5_mean', '6_mean', '7_mean', '8_mean', '9_mean', '10_mean', '11_mean',
         '12_mean', '13_mean', '14_mean', '15_mean', '16_mean', '17_mean', '18_mean', '19_mean', '20_mean', '21_mean',
         '22_mean', '23_mean', '24_mean', '25_mean', '26_mean', '27_mean', '28_mean', '29_mean']]
    labels = df[['label']]
    return np.array(features), np.array(labels)
