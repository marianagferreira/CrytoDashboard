import dash
from dash import dcc
from dash import dash_table
from dash import html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime


#imports for the treemap
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

#!pip install squarify
import matplotlib
import squarify

#!pip install --upgrade plotly
import plotly.graph_objects as go
import plotly.express as px


#########################################
def get_data(token):
    data = yf.Ticker(token)
    df = data.history(period="max")
    df.drop(['Dividends', 'Stock Splits'], inplace=True, axis=1)

    return df

BTC = get_data('BTC-USD')
ADA = get_data('ADA-USD')
ATOM = get_data('ATOM-USD')
AVAX = get_data('AVAX-USD')
AXS = get_data('AXS-USD')
ETH = get_data('ETH-USD')
LINK = get_data('LINK-USD')
LUNA1 = get_data('LUNA1-USD')
MATIC = get_data('MATIC-USD')
SOL = get_data('SOL-USD')
USDT = get_data('USDT-USD')
USDC = get_data('USDC-USD')
BNB = get_data('BNB-USD')
HEX = get_data('HEX-USD')
XRP = get_data('XRP-USD')
BUSD = get_data('BUSD-USD')
DOGE = get_data('DOGE-USD')
DOT = get_data('DOT-USD')
WBTC = get_data('WBTC-USD')
STETH = get_data('STETH-USD')
WTRX = get_data('WTRX-USD')
SHIB = get_data('SHIB-USD')
TRX = get_data('TRX-USD')
DAI = get_data('DAI-USD')
CRO = get_data('CRO-USD')
LTC = get_data('LTC-USD')
NEAR = get_data('NEAR-USD')
LEO = get_data('LEO-USD')
FTT = get_data('FTT-USD')

cryptos = ['BTC', 'ADA', 'ATOM', 'AVAX', 'AXS', 'ETH', 'LINK', 'LUNA1', 'MATIC', 'SOL', 'USDT', 'USDC', 'BNB',
           'HEX', 'XRP', 'BUSD', 'DOGE', 'DOT', 'WBTC', 'STETH', 'SHIB', 'TRX', 'DAI', 'CRO', 'LTC', 'NEAR', 'LEO', 'FTT']

cryptos_dfs =  [BTC, ADA, ATOM, AVAX, AXS, ETH, LINK, LUNA1, MATIC, SOL, USDT, USDC, BNB, HEX, XRP, BUSD, DOGE, DOT, WBTC, STETH,
                SHIB, TRX, DAI, CRO, LTC, NEAR, LEO, FTT]

cryptos_dict = {cryptos[i]: cryptos_dfs[i] for i in range(len(cryptos_dfs))}

#######################################
colors = ['#B6E880', '#FF97FF', '#FECB52', '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']


## treemap
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'10',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': '778da106-dbec-4726-b53c-29e2d98c87bd',
}

session = Session()
session.headers.update(headers)

try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)
except (ConnectionError, Timeout, TooManyRedirects) as e:
    e

    # normalize the data into dataframe format
df = pd.json_normalize(data["data"])
cols_to_keep = ['name', 'symbol', 'cmc_rank', 'quote.USD.price', 'quote.USD.percent_change_24h',
                    'quote.USD.market_cap', ]
df_final = df[cols_to_keep]
    # rename columns
df_final.columns = ['name', 'symbol', 'cmc_rank', 'USD_price', 'USD_percent_change_24h', 'USD_market_cap', ]
    # uncomment below to print the table
    # df_final

######### TABLE WITH CURRENCY PRICES
df_table = pd.DataFrame({

        'BTC': BTC['Close'],
        'ADA': ADA['Close'],
        'ATOM': ATOM['Close'],
        'AVAX': AVAX['Close'],
        'AXS': AXS['Close'],
        'ETH': ETH['Close'],
        'LINK': LINK['Close'],
        'LUNA1': LUNA1['Close'],
        'MATIC': MATIC['Close'],
        'SOL': SOL['Close'],
        'USDT': USDT['Close'],
        'USDC': USDC['Close'],
        'BNB': BNB['Close'],
        'HEX': HEX['Close'],
        'XRP': XRP['Close'],
        'BUSD': BUSD['Close'],
        'DOGE': DOGE['Close'],
        'DOT': DOT['Close'],
        'WBTC': WBTC['Close'],
        'STETH': STETH['Close'],
        'WTRX': WTRX['Close'],
        'SHIB': SHIB['Close'],
        'TRX': TRX['Close'],
        'DAI': DAI['Close'],
        'CRO': CRO['Close'],
        'LTC': LTC['Close'],
        'NEAR': NEAR['Close'],
        'LEO': LEO['Close'],
        'FTT': FTT['Close']
})

result = df_table.tail(2)
result = result.reset_index()
result.drop('Date', axis=1, inplace=True)
result2 = result.assign(Time=['Yesterday $', 'Today $'])
result2 = result2.set_index('Time')
table = result2.transpose()
table["Growth"] = ((table['Yesterday $'] / table['Today $'] - 1) * 100)
table['Growth'] = table['Growth'].round(2)
table['Growth'] = table['Growth'].astype(str) + '%'
table = table.reset_index()
#final_table = table.rename(columns={'index': 'Currency'}, inplace=True)
data_table = pd.DataFrame(table)

# Predictions

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

datelist = pd.date_range(datetime.today(), periods=31).tolist()

#######################################
# Interactive Components
types_of_cryptos = [dict(label=crypto, value=crypto) for crypto in cryptos]

dropdown_types_of_cryptos = dcc.Dropdown(
    id='types_of_cryptos_drop',
    options=types_of_cryptos,
    value='BTC',
    multi=False,  # Start with multiple = False
    clearable = False,
    searchable = False,
    style = {'margin': '4px', 'box-shadow': '0px 0px #ebb36a', 'border-color': '#9F9F5F'}
)

types_of_graphs = ['Prices', 'Candlestick']
types_of_graphs2 = ['Volume', 'Moving Average']

dropdown_types_of_graph = dcc.Dropdown(
    id='types_of_graphs_drop',
    options=types_of_graphs,
    value='Prices',
    multi=False,  # Start with multiple = False
    clearable = False,
    searchable = False,
    style = {'margin': '4px', 'box-shadow': '0px 0px #ebb36a', 'border-color': '#9F9F5F'}
)

dropdown_types_of_graph2 = dcc.Dropdown(
    id='types_of_graphs_drop2',
    options=types_of_graphs2,
    value='Volume',
    multi=False,  # Start with multiple = False
    clearable = False,
    searchable = False,
    style = {'margin': '4px', 'box-shadow': '0px 0px #ebb36a', 'border-color': '#9F9F5F'}
)

year_slider = dcc.RangeSlider(
        id='year_slider',
        min=2014,
        max=2022,
        value = [2014, 2022],
        marks={'2014': 'Year 2014',
                '2015': 'Year 2015',
                '2016': 'Year 2016',
                '2017': 'Year 2017',
                '2018': 'Year 2018',
                '2019': 'Year 2019',
                '2020': 'Year 2020',
                '2021': 'Year 2021',
                '2022': 'Year 2022'},
    tooltip={"placement": "bottom", "always_visible": False},
    step=1
    )

##################################################
# APP
app = dash.Dash(__name__)

server = app.server

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  LAYOUT   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
app.layout = html.Div([

    html.Div([

        # Divide horizontally
        ##############################################################
        html.Div([  # create a box in the assets, for the filters !!
            html.Div([  # first part --> dropdown menus (10%)

                html.Div([  # Instructions for the filters
                    html.Br(),
                    html.Label(
                        'CRYPTOCURRENCY DASHBOARD'
                    ),
                ], style={'font-weight': 'bold', 'width': '100%', 'padding-left': '1px'}, className='text'),


                html.Div([  # dropdown type of crypto
                    html.Br(),
                    html.Div([
                        html.Label('Choose the cryto to analyse'),
                    ], style={'font-weight': 'bold', 'padding-bottom': '5px'}),
                    dropdown_types_of_cryptos,
                ], style={'width': '100%', 'padding-right': '10px'}, className='dash-dropdown'),

            ], style={'display': 'inline-block', 'height': '7%'}),

        ], style={'height': '5%', 'width': '106%'}, className='box_filters'),

        html.Div([  # create a box for year filter
            html.Div([
                html.Div([
                    html.Br(),
                    html.Div([
                        html.Label('Year Range'),
                        ], style={'font-weight': 'bold', 'padding-bottom': '5px', 'padding-left' : '40%'}),
                        year_slider,
                    ], style={'font-weight': 'bold', 'height': '100%', 'width': '700%'},
                            className='year_slider'),

            ], style={'display': 'inline-block', 'height': '20%'}),
        ], style={'height': '20%', 'width': '106%', 'padding-left' : '35%'}),

        ##############################################################



        ####################################################################
        html.Div([  # third part --> line charts

            html.Div([  # line chart on the left
                html.Div([
                    html.Div([
                        html.Div([
                            html.Label('Choose the type of graph'),
                        ], style={'font-weight': 'bold', 'padding-bottom': '5px'}),
                        dropdown_types_of_graph,
                        dcc.Graph(id='line_chart'),
                    ], style={'height': '50%'}),

                ], className="box")
            ], style={'width': '50%'}),

            html.Div([  # line chart on the right
                html.Div([
                    html.Div([
                        html.Div([
                            html.Label('Choose the indicator to visualize'),
                        ], style={'font-weight': 'bold', 'padding-bottom': '5px'}),
                        dropdown_types_of_graph2,
                        dcc.Graph(id='line_chart2'),
                    ], style={'height': '50%'}),

                ], className="box")
            ], style={'width': '50%'}),

        ], style={'display': 'flex', 'width' : '107%'}),

        html.Div([  # fouth part --> tree map and table

            html.Div([  # treemap on the left
                html.Div([
                    html.Div([
                        dcc.Graph(id="treemap"),
                    ], style={'height': '50%'}),

                ], className="box")
            ], style={'width': '65%'}),

            html.Div([  # table on the right
                html.Div([
                        dash_table.DataTable(
                            columns=[{"name": i, "id": i} for i in data_table.columns],
                            data=data_table.to_dict('records'),
                            style_data_conditional=[
                                {
                                    'if': {
                                        'column_id': 'Growth',
                                    },
                                    'backgroundColor': 'lightcoral',
                                    'color': 'black'
                                },
                                {
                                    'if': {
                                        'filter_query': '{Growth} > 0.00%',
                                        'column_id': 'Growth'
                                    },
                                    'backgroundColor': 'lightgreen',
                                    'color': 'black'
                                }],)
                ], className="box")

            ], style={'width': '40%',
                 'height': '490px',
                 'overflow': 'scroll',
                 'padding': '8px 10px 10px 10px'}
          ),


        ], style={'display': 'flex', 'width': '107%'}),


        ####################################################################
        # Label dividing both of the visualizations

        html.Div([  # fifth part - predictions
            html.Div([
                html.Div([
                    dcc.Graph(id='predictions')]
                )
                ], style={'height': '50%'}),

            ], style={'width': '105%'}, className="box")

        ####################################################################

        ####################################################################

    ], style={'display': 'inline-block', 'height': '65%', 'width': '92%'}, className='main'),

])


##############################
# First Callback
@app.callback(
    [
        Output("line_chart", "figure"),
        Output("line_chart2", "figure")
    ],
    [
        Input("year_slider", "value"),
        Input("types_of_cryptos_drop", "value"),
        Input("types_of_graphs_drop", "value"),
        Input("types_of_graphs_drop2", "value")
    ]
)
# first the inputs, then the states, by order
def plot(year1, crypto, graph1, graph2):
    ########################################################################################
    df = cryptos_dict[crypto]

    # Filter the dataset based on the range slider
    filter_df = df[(df.index.year >= year1[0]) & (df.index.year <= year1[1])]

    ########################################################################################
    # First Visualization: Line Chart

    if graph1 == 'Candlestick':
        fig = go.Figure(data=[go.Candlestick(x=filter_df.index,
                                             open=filter_df['Open'],
                                             high=filter_df['High'],
                                             low=filter_df['Low'],
                                             close=filter_df['Close'])])
        fig.update_layout(title_text= crypto + ' Candlestick', title_x=0.5, plot_bgcolor='#F5F5DC')

    else:
        fig = px.line(filter_df, x=filter_df.index, y=filter_df.columns[1:4], title=crypto)
        fig.update_layout(title_text= crypto + ' Price Values', title_x=0.5, plot_bgcolor='#F5F5DC')


    ########################################################################################

    if graph2 == 'Volume':
        fig2 = px.line(filter_df, x=filter_df.index, y=filter_df.Volume, title=crypto)
        fig2.update_layout(title_text= crypto + ' Volume', title_x=0.5, plot_bgcolor='#F5F5DC')

    else:
        # 100 days Moving Averages
        MA100 = filter_df['Open'].rolling(100).mean()  # creat column of 100 day MA
        fig2 = go.Scatter(y=MA100, x=filter_df.index, name=crypto + ' Moving Average')

    return [go.Figure(data = fig), go.Figure(data = fig2)]


#######################################
# Second Callback
@app.callback(
    Output('treemap', 'figure'),
    Input('treemap', 'figure'))

def treemap(value):
    fig_tree = px.treemap(df_final,
                     path=['name'],
                     values='USD_market_cap',
                     color_continuous_scale='RdYlGn',
                     color='USD_percent_change_24h',
                     )
    fig_tree.update_layout(title_text='Cryptomarket price change last 24 hours', title_x=0.5)
    return fig_tree

#######################################
# Third Callback

@app.callback(
    Output('predictions', 'figure'),
    Input('types_of_cryptos_drop', 'value'))

def predictions(crypto):

    df = cryptos_dict[crypto]

    split_point = len(df) - 30
    dataset, validation = df[0:split_point], df[split_point:]
    dataset.to_csv('dataset.csv', index=False)
    validation.to_csv('validation.csv', index=False)

    # seasonal difference
    X = df['Close'].values
    days_test = 30
    differenced = difference(X, days_test)

    # fit model
    model = ARIMA(differenced, order=(7, 0, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast()[0]
    # invert the differenced forecast to something usable
    forecast = inverse_difference(X, forecast, days_test)

    # multi-step out-of-sample forecast
    start_index = len(differenced)
    end_index = start_index + 30
    forecast = model_fit.predict(start=start_index, end=end_index)

    # invert the differenced forecast to something usable
    history = [x for x in X]
    day = 1
    predictions = list()
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_test)
        # print('Day %d: %f' % (day, inverted))
        history.append(inverted)
        day += 1

        predictions.append(float(inverted))

    crypto_pred = px.line(x=datelist, y=predictions, title=crypto + ' Predicted close price for the next 30 days')
    crypto_pred.update_layout(title_x=0.5, plot_bgcolor='#F5F5DC', yaxis_title = 'Close price', xaxis_title = 'Date')

    return go.Figure(data=crypto_pred)

#######################################


############################################################
if __name__ == '__main__':
    app.run_server(debug=True)
