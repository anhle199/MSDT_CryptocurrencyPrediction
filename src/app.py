from os.path import isfile

from dash import Dash, Patch, callback, dcc, html
from dash.dependencies import Input, Output
from keras.models import load_model
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

from prediction import *


CRYPTO_LIST = ['BTC', 'ETH', 'ADA']
DATASETS = {}


def load_data():
    for crypto_name in CRYPTO_LIST:
        dataset = prepare_dataset(crypto_name)
        DATASETS[crypto_name] = dataset


def predict(name):
    dataset = DATASETS[name]
    train_data, valid_data = split_dataset(dataset)

    model_filename = build_model_filename(name)
    scaler = MinMaxScaler(feature_range=(0, 1))

    if isfile(model_filename):
        # use prebuilt model
        scaler.fit(train_data.values)
        model = load_model(model_filename)
    else:
        # normalize dataset and train lstm model
        x_train_data, y_train_data = normalize_dataset(dataset, scaler)
        model = train_lstm_model(x_train_data, y_train_data)
        model.save(model_filename)

    # predict closing price
    inputs = get_sample_dataset(dataset, scaler)
    closing_price = predict_close_price(model, scaler, inputs)

    valid_data['Predictions'] = closing_price
    return valid_data


load_data()

app = Dash()
server = app.server


app.layout = html.Div(
    [
        html.H1('Cryptocurrency Price Analysis Dashboard', style={'textAlign': 'center'}),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id='crypto-prediction-dropdown',
                            options=[
                                {'label': 'Bitcoin', 'value': 'BTC'},
                                {'label': 'Ethereum', 'value': 'ETH'},
                                {'label': 'Cardano', 'value': 'ADA'},
                            ],
                            value='BTC',
                            style={'width': '100%', 'margin-right': '12px'},
                        ),
                        html.Button('Refresh', id='refresh-button'),
                    ],
                    style={'display': 'flex'},
                ),
                html.H2('Actual closing price', style={'textAlign': 'center'}),
                dcc.Graph(
                    id='actual-closing-price-chart',
                    figure={
                        'data': [go.Scatter(mode='markers')],
                    },
                ),
                html.H2('LSTM Predicted closing price', style={'textAlign': 'center'}),
                dcc.Graph(
                    id='predicted-closing-price-chart',
                    figure={
                        'data': [go.Scatter(mode='markers')],
                    },
                ),
            ]
        ),
    ],
)


@callback(
    Output('actual-closing-price-chart', 'figure'),
    Output('predicted-closing-price-chart', 'figure'),
    Input('crypto-prediction-dropdown', 'value'),
    Input('refresh-button', 'n_clicks'),
)
def update_graph(selected_crypto, n_clicks):
    print(f'>>> triggered callback, {selected_crypto}, {n_clicks}')
    if n_clicks is not None and n_clicks > 0:
        load_data()
        print(f'>>> reloaded data successfully')

    valid_data = predict(selected_crypto)

    patched_actual_figure = Patch()
    patched_actual_figure['data'][0]['x'] = valid_data.index
    patched_actual_figure['data'][0]['y'] = valid_data.Close

    patched_predicted_figure = Patch()
    patched_predicted_figure['data'][0]['x'] = valid_data.index
    patched_predicted_figure['data'][0]['y'] = valid_data.Predictions

    return [patched_actual_figure, patched_predicted_figure]


if __name__ == '__main__':
    app.run(debug=True)
