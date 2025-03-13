import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta


class RealWorldCommodityTrader:
    def __init__(self, commodity_symbol, start_date='2010-01-01', initial_capital=10000):
        self.commodity_symbol = commodity_symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        self.backtesting_start = self.end_date - timedelta(days=2 * 365)
        self.commodity_data = self.fetch_data(commodity_symbol)
        self.economic_indicators = self.fetch_economic_indicators()
        self.model = self.build_lstm_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0

    def fetch_data(self, symbol):
        data = yf.download(symbol, start=self.start_date, end=self.end_date)
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]

    def fetch_economic_indicators(self):
        indicators = {
            'SPY': 'S&P500',
            'TLT': 'Long_Term_Bonds',
            'GLD': 'Gold',
            'DBC': 'Commodities',
            'UUP': 'US_Dollar',
            '^FTSE': 'UK_FTSE',
            'AEX.AS': 'Netherlands_AEX',
            '000001.SS': 'China_SSE'
        }
        data = pd.DataFrame()
        for symbol, name in indicators.items():
            df = self.fetch_data(symbol)['Close']
            df.name = name
            data = pd.concat([data, df], axis=1)
        return data

    def prepare_data(self):
        combined_data = pd.concat([
            self.commodity_data,
            self.economic_indicators
        ], axis=1).dropna()

        training_data = combined_data[combined_data.index < self.backtesting_start]

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'S&P500',
                    'Long_Term_Bonds', 'Gold', 'Commodities', 'US_Dollar',
                    'UK_FTSE', 'Netherlands_AEX', 'China_SSE']

        scaled_data = self.scaler.fit_transform(training_data[features])

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i - 60:i])
            y.append(scaled_data[i, 3])  # predict Close price
        return np.array(X), np.array(y)

    def build_lstm_model(self):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(60, 13)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train_model(self):
        X, y = self.prepare_data()
        self.model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    def predict_price(self, input_data):
        scaled_input = self.scaler.transform(input_data)
        scaled_prediction = self.model.predict(np.array([scaled_input[-60:]]))
        return self.scaler.inverse_transform(scaled_prediction)[0][0]

    def calculate_technical_indicators(self, data):
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        return data

    def make_trading_decision(self, current_data, predicted_price):
        current_price = current_data['Close'].iloc[-1]
        sma_20 = current_data['SMA_20'].iloc[-1]
        ema_20 = current_data['EMA_20'].iloc[-1]
        rsi = current_data['RSI'].iloc[-1]
        macd = current_data['MACD'].iloc[-1]
        signal_line = current_data['Signal_Line'].iloc[-1]

        if (predicted_price > current_price * 1.05 and
                current_price > sma_20 and
                current_price > ema_20 and
                rsi < 70 and
                macd > signal_line):
            return "STRONG BUY"
        elif (predicted_price > current_price * 1.02 and
              (current_price > sma_20 or current_price > ema_20) and
              rsi < 60):
            return "BUY"
        elif (predicted_price < current_price * 0.95 and
              current_price < sma_20 and
              current_price < ema_20 and
              rsi > 30 and
              macd < signal_line):
            return "STRONG SELL"
        elif (predicted_price < current_price * 0.98 and
              (current_price < sma_20 or current_price < ema_20) and
              rsi > 40):
            return "SELL"
        else:
            return "HOLD"

    def execute_trade(self, decision, price):
        if decision in ["STRONG BUY", "BUY"] and self.position == 0:
            self.position = self.current_capital / price
            self.current_capital = 0
        elif decision in ["STRONG SELL", "SELL"] and self.position > 0:
            self.current_capital = self.position * price
            self.position = 0

    def backtest(self):
        self.train_model()

        backtesting_data = pd.concat([
            self.commodity_data,
            self.economic_indicators
        ], axis=1).dropna()

        backtesting_data = backtesting_data[backtesting_data.index >= self.backtesting_start]
        backtesting_data = self.calculate_technical_indicators(backtesting_data)

        for i in range(60, len(backtesting_data)):
            input_data = backtesting_data.iloc[i - 60:i]
            current_price = backtesting_data['Close'].iloc[i]

            predicted_price = self.predict_price(input_data)
            decision = self.make_trading_decision(input_data, predicted_price)

            self.execute_trade(decision, current_price)

        final_value = self.current_capital + (self.position * backtesting_data['Close'].iloc[-1])
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        print(f"Initial Capital: €{self.initial_capital:.2f}")
        print(f"Final Value: €{final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Return: {((1 + total_return / 100) ** (1 / 2) - 1) * 100:.2f}%")


# Usage
trader = RealWorldCommodityTrader('GC=F', initial_capital=10000)  # Gold Futures
trader.backtest()