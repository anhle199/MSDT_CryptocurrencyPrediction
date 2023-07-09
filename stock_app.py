from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from stock_pred import *


def load_live_cryptocurrency_data(names: list[str], currency: str) -> DataFrame:
    start = 0
    df = pd.DataFrame()

    for name in names:
        # download live cryptocurrency dataset
        data = yf.download(tickers=f"{name}-{currency}", period="7d", interval="15m")

        # add "Date" and "Stock" columns
        data_count = len(data)
        data.insert(0, "Date", data.index)
        data.insert(len(data.columns), "Stock", [name] * data_count)

        # set range numeric index
        end = start + data_count
        data.index = pd.RangeIndex(start, end, 1)
        start = end

        df = pd.concat([df, data])

    return df


app = Dash()
server = app.server

# build dataset
# dataset = pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
dataset = load_live_cryptocurrency_data(["BTC"], "USD")
dataset["Date"] = pd.to_datetime(dataset.Date, format="%Y-%m-%d")
sorted_dataset = dataset.sort_values(by="Date", ascending=True, axis=0)
filtered_dataset = pd.DataFrame(data=sorted_dataset.Close.to_numpy(), index=sorted_dataset.Date, columns=["Close"])

scaler = MinMaxScaler(feature_range=(0, 1))
# normalize dataset and train lstm model
x_train_data, y_train_data = normalize_dataset(filtered_dataset, scaler)
model = train_lstm_model(x_train_data, y_train_data)

# predict close price
inputs = get_sample_dataset(filtered_dataset, scaler)
closing_price = predict_close_price(model, scaler, inputs)

train_data, valid_data = split_dataset(filtered_dataset)
print('total - train - valid', len(filtered_dataset), len(train_data), len(valid_data))
valid_data["Predictions"] = closing_price

df = load_live_cryptocurrency_data(["BTC", "ETH", "ADA"], "USD")

app.layout = html.Div(
    [
        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="Cryptocurrency Stock Data",
                    children=[
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="prediction-dropdown",
                                    options=[
                                        {"label": "Bitcoin", "value": "BTC"},
                                        {"label": "Ethereum", "value": "ETH"},
                                        {"label": "Cardano", "value": "ADA"},
                                    ],
                                    multi=False,
                                    value=["BTC"],
                                ),
                                html.H2("Actual closing price", style={"textAlign": "center"}),
                                dcc.Graph(
                                    id="Actual Data",
                                    figure={
                                        "data": [go.Scatter(x=valid_data.index, y=valid_data["Close"], mode="markers")],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                                dcc.Graph(
                                    id="Predicted Data",
                                    figure={
                                        "data": [
                                            go.Scatter(x=valid_data.index, y=valid_data["Predictions"], mode="markers")
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Bitcoin Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H1("Bitcoin Stocks High vs Lows", style={"textAlign": "center"}),
                                dcc.Dropdown(
                                    id="my-dropdown",
                                    options=[
                                        {"label": "Bitcoin", "value": "BTC"},
                                        {"label": "Ethereum", "value": "ETH"},
                                        {"label": "Cardano", "value": "ADA"},
                                    ],
                                    multi=True,
                                    value=["BTC"],
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="highlow"),
                                html.H1("Bitcoin Market Volume", style={"textAlign": "center"}),
                                dcc.Dropdown(
                                    id="my-dropdown2",
                                    options=[
                                        {"label": "Bitcoin", "value": "BTC"},
                                        {"label": "Ethereum", "value": "ETH"},
                                        {"label": "Cardano", "value": "ADA"},
                                    ],
                                    multi=True,
                                    value=["BTC"],
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="volume"),
                            ],
                            className="container",
                        ),
                    ],
                ),
            ],
        ),
    ]
)

# @app.callback([Output("Actual data", "figure"), Output("Predicted data", "figure")], [Input("prediction-dropdown", "value")])
# def update_graph(selected_dropdown):

@app.callback(Output("highlow", "figure"), [Input("my-dropdown", "value")])
def update_graph(selected_dropdown):
    dropdown = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "ADA": "Cardano",
    }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["High"],
                mode="lines",
                opacity=0.7,
                name=f"High {dropdown[stock]}",
                textposition="bottom center",
            )
        )
        trace2.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["Low"],
                mode="lines",
                opacity=0.6,
                name=f"Low {dropdown[stock]}",
                textposition="bottom center",
            )
        )
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={
                "title": "Date",
                "rangeselector": {
                    "buttons": list(
                        [
                            {"count": 7, "label": "7 days", "step": "day", "stepmode": "backward"},
                            {"count": 1, "label": "1 month", "step": "month", "stepmode": "backward"},
                            {"step": "all"},
                        ]
                    )
                },
                "rangeslider": {"visible": True},
                "type": "date",
            },
            yaxis={"title": "Price (USD)"},
        ),
    }
    return figure


@app.callback(Output("volume", "figure"), [Input("my-dropdown2", "value")])
def update_graph(selected_dropdown_value):
    dropdown = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "ADA": "Cardano",
    }
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["Volume"],
                mode="lines",
                opacity=0.7,
                name=f"Volume {dropdown[stock]}",
                textposition="bottom center",
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={
                "title": "Date",
                "rangeselector": {
                    "buttons": list(
                        [
                            {"count": 7, "label": "7 days", "step": "day", "stepmode": "backward"},
                            {"count": 1, "label": "1 month", "step": "month", "stepmode": "backward"},
                            {"step": "all"},
                        ]
                    )
                },
                "rangeslider": {"visible": True},
                "type": "date",
            },
            yaxis={"title": "Transactions Volume"},
        ),
    }
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
