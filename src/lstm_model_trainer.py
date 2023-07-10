from sklearn.preprocessing import MinMaxScaler

from prediction import *


def build_dataset_filename(crypto_name: str, path: str = DEFAULT_MODEL_DATASET_PATH) -> str:
    prefix = path if path.endswith('/') else path + '/'
    return f'{prefix}{crypto_name}_dataset.csv'


def train_model(dataset: DataFrame) -> Sequential:
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_data, y_train_data = normalize_dataset(dataset, scaler)
    model = train_lstm_model(x_train_data, y_train_data)
    return model


if __name__ == '__main__':
    for crypto_name in ['BTC', 'ETH', 'ADA']:
        print(f'>>> Cryptocurrency: {crypto_name}')

        # prepare dataset
        print(f'>>> preparing dataset...')
        dataset = prepare_dataset(crypto_name)
        dataset.to_csv(build_dataset_filename(crypto_name))
        print(f'>>> prepared dataset successfully')

        print()

        # train model
        print(f'>>> training LSTM model...')
        model = train_model(dataset)
        model.save(build_model_filename(crypto_name))
        print(f'>>> trained LSTM model successfully')

        print()
