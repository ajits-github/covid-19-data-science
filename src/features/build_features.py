import numpy as np
from sklearn import linear_model

reg = linear_model.LinearRegression(fit_intercept=True)
import pandas as pd

from scipy import signal

def get_doubling_time(in_array):
    # Using a linear regression to approximate the doubling rate

    y = np.array(in_array)
    X = np.arange(-1, 2).reshape(-1, 1)

    assert len(in_array) == 3
    reg.fit(X, y)
    intercept = reg.intercept_
    slope = reg.coef_

    return intercept / slope


def savgol_filter(df_input, column='confirmed', window=5):
    df_result = df_input

    filter_in = df_input[column].fillna(0)

    result = signal.savgol_filter(np.array(filter_in),
                                  window,
                                  1)
    df_result[str(column + '_filtered')] = result
    return df_result


def rolling_regression(df_input, col='confirmed'):
    # Rolling Regression to approximate the doubling time

    days_back = 3
    result = df_input[col].rolling(
        window=days_back,
        min_periods=days_back).apply(get_doubling_time, raw=False)

    return result


def calculate_filtered_data(df_input, filter_on='confirmed'):
    #  Calculate savgol filter and return merged data frame

    must_contain = set(['state', 'country', filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    df_output = df_input.copy()  # making a copy here so that the column filter_on  don't get be overwritten

    pd_filtered_result = df_output[['state', 'country', filter_on]].groupby(['state', 'country']).apply(
        savgol_filter)  # .reset_index()

    df_output = pd.merge(df_output, pd_filtered_result[[str(filter_on + '_filtered')]], left_index=True,
                         right_index=True, how='left')
    return df_output.copy()


def calculate_doubling_rate(df_input, filter_on='confirmed'):
    # Calculate approximated doubling rate and return merged data frame

    must_contain = set(['state', 'country', filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    pd_DR_result = df_input.groupby(['state', 'country']).apply(rolling_regression, filter_on).reset_index()

    pd_DR_result = pd_DR_result.rename(columns={filter_on: filter_on + '_DR',
                                                'level_2': 'index'})

    # we do the merge on the index of our big table and on the index column after groupby
    df_output = pd.merge(df_input, pd_DR_result[['index', str(filter_on + '_DR')]], left_index=True, right_on=['index'],
                         how='left')
    df_output = df_output.drop(columns=['index'])

    return df_output


if __name__ == '__main__':
    test_data_reg = np.array([2, 4, 6])
    result = get_doubling_time(test_data_reg)
    print('the test slope is: ' + str(result))

    pd_JH_data = pd.read_csv('data/processed/COVID_relational_confirmed.csv', sep=';', parse_dates=[0])
    pd_JH_data = pd_JH_data.sort_values('date', ascending=True).copy()

    pd_result_large = calculate_filtered_data(pd_JH_data)
    pd_result_large = calculate_doubling_rate(pd_result_large)
    pd_result_large = calculate_doubling_rate(pd_result_large, 'confirmed_filtered')

    mask = pd_result_large['confirmed'] > 100
    pd_result_large['confirmed_filtered_DR'] = pd_result_large['confirmed_filtered_DR'].where(mask, other=np.NaN)
    pd_result_large.to_csv('data/processed/COVID_final_set.csv', sep=';', index=False)
    print(pd_result_large[pd_result_large['country'] == 'Germany'].tail())
