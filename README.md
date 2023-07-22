# CRYPTOCURRENCY PRICE PREDICTION

## Install dependencies
- Run this command at the root directory:
  ```
  pip install -r requirements.txt
  ```

## Runs app
#### 1. Using source code
- Go to `src` directory:
  ```
  cd src
  ```
- Runs app on local (using `python3` if your machine has both python 2 and python 3):
  ```
  python app.py
  ```
- App is running on `http://localhost:8050`, you can view it in the browser.

#### 2. Using Docker
- Run below commands at the root directory.
- Build docker image:
  - For compose v1:
    ```
    docker-compose -f docker-compose.build.yml build crypto_prediction
    ```
  - For compose v2:
    ```
    ./build.sh
    ```
    or
    ```
    docker-compose -f docker-compose.build.yml build crypto_prediction
    ```
- Create container and run app:
  - For compose v1:
    ```
    docker-compose compose up -d crypto_prediction
    ```
  - For compose v2:
    ```
    docker compose compose up -d crypto_prediction
    ```
  - App is running on `http://localhost:8050`, you can view it in the browser.

## Video demo
- [[CSC13115 - 19KTPM] Cryptocurrency Prediction App Demo](https://youtu.be/SluTbyecGxg)

## References
- [Stock price predictino machine learning project in Python](https://data-flair.training/blogs/stock-price-prediction-machine-learning-project-in-python/?fbclid=IwAR32s4iEGVLbwDhlVnvxRbFyNvL6XHVQCXWTP9wrEzVKPOtI65WnmZZa4rk)
