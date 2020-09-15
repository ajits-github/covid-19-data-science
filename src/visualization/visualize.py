import pandas as pd
import numpy as np
import subprocess
import dash

dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go

from sklearn import linear_model

reg = linear_model.LinearRegression(fit_intercept=True)

from scipy import signal
import os

from src.data.get_data import get_johns_hopkins
from src.data.process_JH_data import store_relational_data
from src.features.build_features import *
from src.data.get_pd_large import pd_result_large
get_johns_hopkins()
store_relational_data()
pd_result_large()
print(os.getcwd())
df_input_large = pd.read_csv('../../data/processed/COVID_final_set.csv', sep=';')

fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  #  Applied Data Science on COVID-19 data
    

    '''),

    dcc.Markdown('''
    ## Multi-option Country for visualization
    '''),

    dcc.Dropdown(
        id='country_drop_down',
        options=[{'label': each, 'value': each} for each in df_input_large['country'].unique()],
        value=['Kuwait', 'Germany', 'United Kingdom'],  # which are pre-selected
        multi=True
    ),

    dcc.Markdown('''
        ## Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        '''),

    dcc.Dropdown(
        id='doubling_time',
        options=[
            {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
            {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
            {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
            {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
        ],
        value='confirmed',
        multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope')
])


@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
     Input('doubling_time', 'value')])
def update_figure(country_list, show_doubling):
    if 'doubling_rate' in show_doubling:
        my_yaxis = {'type': "log",
                    'title': 'Approximated doubling rate over 3 days (hint: large numbers are better)'
                    }
    else:
        my_yaxis = {'type': "log",
                    'title': 'Confirmed infected people (source johns hopkins csse, log-scale)'
                    }

    traces = []
    for each in country_list:

        df_plot = df_input_large[df_input_large['country'] == each]

        if show_doubling == 'doubling_rate_filtered':
            df_plot = df_plot[
                ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                 'date']].groupby(['country', 'date']).agg(np.mean).reset_index()
        else:
            df_plot = df_plot[
                ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                 'date']].groupby(['country', 'date']).agg(np.sum).reset_index()

        traces.append(dict(x=df_plot.date,
                           y=df_plot[show_doubling],
                           mode='markers+lines',
                           opacity=0.9,
                           name=each
                           )
                      )

    return {
        'data': traces,
        'layout': dict(
            width=1280,
            height=720,

            xaxis={'title': 'Timeline',
                   'tickangle': -45,
                   'nticks': 20,
                   'tickfont': dict(size=14, color="#7f7f7f"),
                   },

            yaxis=my_yaxis
        )
    }


def get_doubling_time(array):
    # Use a linear regression to approximate the doubling rate

    y = np.array(array)
    x = np.arange(-1, 2).reshape(-1, 1)

    assert len(array) == 3
    reg.fit(x, y)
    intercept = reg.intercept_
    slope = reg.coef_

    return intercept / slope


def savgol_filter(df_input, column='confirmed', window=5):
    # Savgol Filter which can be used in groupby apply function

    df_result = df_input
    filter_in = df_input[column].fillna(0)

    result = signal.savgol_filter(np.array(filter_in),
                                  window,
                                  1)
    df_result[str(column + '_filtered')] = result
    return df_result


def rolling_regression(df_input, col='confirmed'):
    # Rolling Regression to approximate the doubling time'

    days_back = 3
    result = df_input[col].rolling(
        window=days_back,
        min_periods=days_back).apply(get_doubling_time, raw=False)

    return result


def calc_filtered_data(df_input, filter_on='confirmed'):
    # Calculate savgol filter and return merged data frame

    must_contain = set(['state', 'country', filter_on])
    assert must_contain.issubset(set(df_input.columns)),'Error in calculate_filtered_data not all columns in data frame'

    df_output = df_input.copy()

    pd_filtered_result = df_output[['state', 'country', filter_on]].groupby(['state', 'country']).apply(
        savgol_filter)  # .reset_index()

    df_output = pd.merge(df_output, pd_filtered_result[[str(filter_on + '_filtered')]], left_index=True,
                         right_index=True, how='left')
    return df_output.copy()


def calc_doubling_rate(df_input, filter_on='confirmed'):
    # Calculate approximated doubling rate and return merged data frame

    must_contain = set(['state', 'country', filter_on])
    assert must_contain.issubset(set(df_input.columns)),'Error in calculate_filtered_data not all columns in data frame'

    pd_DR_result = df_input.groupby(['state', 'country']).apply(rolling_regression, filter_on).reset_index()

    pd_DR_result = pd_DR_result.rename(columns={filter_on: filter_on + '_DR',
                                                'level_2': 'index'})

    df_output = pd.merge(df_input, pd_DR_result[['index', str(filter_on + '_DR')]], left_index=True, right_on=['index'],
                         how='left')
    df_output = df_output.drop(columns=['index'])

    return df_output


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
