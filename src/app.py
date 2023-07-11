import os

from dash import Dash, Patch, callback, dcc, html
from dash.dependencies import Input, Output
from keras.models import load_model
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

from prediction import *


def predict(name):
    dataset = prepare_dataset(name, '1d', '1m')
    train_data, valid_data = split_dataset(dataset)

    model_filename = build_model_filename(name)
    scaler = MinMaxScaler(feature_range=(0, 1))

    if os.path.isfile(model_filename):
        # use prebuilt model
        scaler.fit(train_data.values)
        model = load_model(model_filename)
    else:
        # normalize dataset and train lstm model
        train_dataset = prepare_dataset(name, '7d', '15m')
        x_train_data, y_train_data = normalize_dataset(train_dataset, scaler)
        model = train_lstm_model(x_train_data, y_train_data)
        model.save(model_filename)

    # predict closing price
    inputs = get_sample_dataset(dataset, scaler)
    closing_price = predict_close_price(model, scaler, inputs)

    valid_data['Predictions'] = closing_price
    return valid_data


debug = os.environ.get("DASH_DEBUG_MODE", "True") == "True"
app = Dash()
server = app.server

app.layout = html.Div(
    [
        html.H1('Cryptocurrency Price Prediction Dashboard', style={'textAlign': 'center'}),
        html.Div(
            [
                dcc.Dropdown(
                    id='crypto-dropdown',
                    options=[
                        {'label': 'Bitcoin', 'value': 'BTC'},
                        {'label': 'Ethereum', 'value': 'ETH'},
                        {'label': 'Cardano', 'value': 'ADA'},
                    ],
                    value='BTC',
                    clearable=False,
                ),
                dcc.Loading(
                    id='loading-graphs',
                    type='circle',
                    children=[
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
                    ],
                ),
            ]
        ),
    ],
)


@callback(
    Output('actual-closing-price-chart', 'figure'),
    Output('predicted-closing-price-chart', 'figure'),
    Input('crypto-dropdown', 'value'),
)
def update_graph(selected_crypto):
    valid_data = predict(selected_crypto)

    patched_actual_figure = Patch()
    patched_actual_figure['data'][0]['x'] = valid_data.index
    patched_actual_figure['data'][0]['y'] = valid_data.Close

    patched_predicted_figure = Patch()
    patched_predicted_figure['data'][0]['x'] = valid_data.index
    patched_predicted_figure['data'][0]['y'] = valid_data.Predictions

    return [patched_actual_figure, patched_predicted_figure]


if __name__ == '__main__':
    app.run(debug=debug)
