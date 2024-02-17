# Test TS which can use
# https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data?select=sample_submission.csv

# TODO:
# fix plot_components
# fix in to_vanga add draw_forecast
# fix draw_table_of_future_trend without gridSearch
# fix module_pocket
# add method - validation loss
# add Decompose
# add Anomalys

# Standart
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from tqdm.auto import tqdm

# metrics & gridSearch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid

import random 

# Model
from prophet import Prophet

# Graphics
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

# Graphics size
from pylab import rcParams
rcParams['figure.figsize'] = 15, 7

class time_series:    
    def __init__(self, df, interval = 90, test_size = 30, interval_width = 0.95, loss = 'MAPE'):
        self.data = df.copy()
        self.data['ds'] = pd.to_datetime(self.data['ds'])
        self.interval = interval
        self.test_size = test_size
        self.interval_width = interval_width
        self.loss = loss
        
        self.data_train = self.data.iloc[: -test_size] 
        self.data_test = self.data.iloc[-test_size:]
        
        self.parametres_dict = {}
        self.losses_with_parameters_table = None
    
    # ============ Setters ============
    def set_inteval(self, interval):
        self.interval = interval

    def set_test_size(self, test_size):
        self.test_size = test_size
        self.data_train = self.data.iloc[:-test_size] 
        self.data_test = self.data.iloc[-test_size:]
        
    def set_interval_width(self, interval_width):
        self.interval_width = interval_width
        
    def set_loss(self, loss):
        self.loss = loss
        
    def set_custom_test_size(self, date_test_start):
        date_test_start = pd.Timestamp(date_test_start)
        self.data_train = self.data[(self.data.ds < date_test_start)]
        self.data_test = self.data[(self.data.ds >= date_test_start)]
        self.test_size = len(self.data_test)
        
    def set_model_parameters(self, dict_of_params):
        for param in dict_of_params:
            self.parametres_dict[param] = dict_of_params[param]
    
    # ============ Getters ============
    
    
    
    # ============ Model ============    
    def search_params(self, changepoint_prior_scale = [0.01, 0.05, 0.1], 
                            seasonality_prior_scale = [0.01, 0.1, 1, 10],
                            holidays_prior_scale = [0.01, 0.1, 1, 10],
                            seasonality_mode = ['additive', 'multiplicative'],
                            changepoint_range = [0.8, 0.9, 0.95]):
        
        params_grid = {'changepoint_prior_scale': changepoint_prior_scale,
                       'seasonality_prior_scale': seasonality_prior_scale,
                       'holidays_prior_scale': holidays_prior_scale,
                       'seasonality_mode': seasonality_mode,
                       'changepoint_range': changepoint_range}
        grid = ParameterGrid(params_grid)
        
        model_parameters = pd.DataFrame(columns = ['MSE', 'MAE', 'MAPE', 
                                                   'changepoint_prior_scale', 'seasonality_prior_scale',
                                                   'holidays_prior_scale', 'seasonality_mode', 'changepoint_range'])
        for p in tqdm(grid):
            test = pd.DataFrame()
            random.seed(0)
            
            train_model = Prophet(interval_width=self.interval_width,
                                 changepoint_prior_scale = p['changepoint_prior_scale'],
                                 seasonality_prior_scale = p['seasonality_prior_scale'],
                                 holidays_prior_scale = p['holidays_prior_scale'],
                                 seasonality_mode = p['seasonality_mode'],
                                 changepoint_range = p['changepoint_range']
                                )
            train_model.add_country_holidays(country_name='Russia')
            train_model.fit(self.data_train)
            
            train_forecast = train_model.make_future_dataframe(periods=self.test_size, freq='D', include_history=False)
            train_forecast = train_model.predict(train_forecast)
            
            test = train_forecast[['ds','yhat']]
            Actual = self.data_test
            
            MSE = mean_squared_error(Actual['y'],test['yhat'])
            MAE = mean_absolute_error(Actual['y'],test['yhat'])
            MAPE = mean_absolute_percentage_error(Actual['y'],abs(test['yhat']))
            
            model_parameters = pd.concat([model_parameters, pd.DataFrame({'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE, 
                                                                          'changepoint_prior_scale': p['changepoint_prior_scale'],
                                                                          'seasonality_prior_scale': p['seasonality_prior_scale'],
                                                                          'holidays_prior_scale': p['holidays_prior_scale'],
                                                                          'seasonality_mode': p['seasonality_mode'],
                                                                          'changepoint_range': p['changepoint_range']}
                                                                        , index=[0])]).reset_index(drop=True)

        self.losses_with_parameters_table = model_parameters.sort_values(by=[self.loss]).reset_index(drop=True)
        
        self.parametres_dict = {
            'changepoint_prior_scale': self.losses_with_parameters_table.changepoint_prior_scale[0],
            'seasonality_prior_scale': self.losses_with_parameters_table.seasonality_prior_scale[0],
            'holidays_prior_scale':    self.losses_with_parameters_table.holidays_prior_scale[0],
            'seasonality_mode':        self.losses_with_parameters_table.seasonality_mode[0],
            'changepoint_range':       self.losses_with_parameters_table.changepoint_range[0]
        }
        
        print('Ошибки на валидации:')
        print(f"MSE = {self.losses_with_parameters_table['MSE'][0]}")
        print(f"MAE = {self.losses_with_parameters_table['MAE'][0]}")
        print(f"MAPE = {self.losses_with_parameters_table['MAPE'][0]}")
        print('Текущие параметры после gridSearch:', end='\n\n')
        for param in self.parametres_dict:
            print(f'{param} = {self.parametres_dict[param]}')
        
    def init_and_predict_prophet(self):
        self.prophet_model = Prophet(interval_width = self.interval_width,
                                     changepoint_prior_scale = self.parametres_dict['changepoint_prior_scale'],
                                     seasonality_prior_scale = self.parametres_dict['seasonality_prior_scale'],
                                     holidays_prior_scale = self.parametres_dict['holidays_prior_scale'],
                                     seasonality_mode = self.parametres_dict['seasonality_mode'],
                                     changepoint_range = self.parametres_dict['changepoint_range']
                               )
        self.prophet_model.add_country_holidays(country_name='Russia')

        # Обучаем
        self.prophet_model.fit(self.data)
        # Выдаем предикт 
        future_df = self.prophet_model.make_future_dataframe(periods=self.interval, freq='D', include_history=True)
        self.forecast = self.prophet_model.predict(future_df)
        
    def plot_components(self):
        self.prophet_model.plot_components(self.forecast);
    
    # ============ Draw graphics ============
    def draw_timeseries(self):
        plt.title('Реальная метрика')
        plt.plot(self.data['ds'].to_numpy(), self.data['y'].to_numpy(), label='real')
        plt.show()
        
    def draw_test_and_train_dataset(self):
        plt.title(f'График test and train')
        plt.plot(self.data_train['ds'].to_numpy(), self.data_train['y'].to_numpy(), label='train')
        plt.plot(self.data_test['ds'].to_numpy(), self.data_test['y'].to_numpy(), label='test')
        plt.legend()
        plt.show()
        
    def draw_forecast(self):
        plt.title(f'График для будущего')
        plt.plot(self.data['ds'].to_numpy(), self.data['y'].to_numpy(), label='real')
        plt.plot(self.forecast['ds'].to_numpy(), self.forecast['yhat'].to_numpy(), label='forecast')
        plt.fill_between(self.forecast['ds'].to_numpy(), 
                         self.forecast['yhat_lower'].to_numpy(), self.forecast['yhat_upper'].to_numpy(), 
                         alpha=0.3, color='grey', label=f'дов. интервал {self.interval_width}')
        plt.legend()
        plt.show()
        
    def draw_trend(self):
        plt.title('График для тренда на будущее')
        plt.plot(self.forecast['ds'].to_numpy(), self.forecast['trend_upper'].to_numpy(), label='upper')
        plt.plot(self.forecast['ds'].to_numpy(), self.forecast['trend_lower'].to_numpy(), label='lower')
        plt.fill_between(self.forecast['ds'].to_numpy(), 
                         self.forecast['trend_lower'].to_numpy(), self.forecast['trend_upper'].to_numpy(), 
                         alpha=0.5, color='grey', label=f'дов. интервал {self.interval_width}')
        plt.legend()
        plt.show()
        
    def draw_table_of_future_trend(self, future_quater=None):
        if self.losses_with_parameters_table is None:
            print('Выдать ожидаемый прирост исходя из ошибки на валидации сейчас нельзя, так как надо посмотреть на ошибки на валидации. Запустить метод гридсерча или ошибки на валидации.')
            return

        MAPE = self.losses_with_parameters_table['MAPE'][0]
        
        if future_quater is None: 
            future_quater = str(pd.to_datetime(self.data.ds.max()).year) + 'Q' + str(pd.to_datetime(self.data.ds.max()).quarter)
        
        # Подготовить табличку для того, чтоб выдать ожидаемый прирост с ошибкой
        res_growth = self.forecast.copy()
        res_growth['quater'] = pd.PeriodIndex(res_growth['ds'], freq='Q')
        res_growth = res_growth[res_growth.quater <= future_quater]
        res_growth = res_growth.groupby('quater', as_index=False)[['trend_lower', 'trend_upper']].mean()

        res_growth['shift_trend_lower'] = res_growth['trend_lower'].shift()
        res_growth['shift_trend_upper'] = res_growth['trend_upper'].shift()

        res_growth['diff_trend_lower'] = res_growth['trend_lower'] / res_growth['shift_trend_lower'] - 1
        res_growth['diff_trend_upper'] = res_growth['trend_upper'] / res_growth['shift_trend_upper'] - 1

        # Выдать ожидаемый прирост исходя из ошибки на валидации (Поправка делается по MAPE)
        print('Ожидаемый прирост в следующем квратале с учетом тренда, исходя из ошибки модели на валидации')
        growth_with_validation = {}
        growth_with_validation['Нижняя граница прогноза'] = round(res_growth.trend_lower.iloc[-1], 4)
        growth_with_validation['Нижняя граница прогноза (с поправкой на ошибку модели)'] = round(res_growth.trend_lower.iloc[-1] * (1 - MAPE) , 4)
        growth_with_validation['Нижняя граница прогноза в %'] = round(res_growth.diff_trend_lower.iloc[-1] * 100, 4)
        growth_with_validation['Нижняя граница прогноза (с поправкой на ошибку модели) в %'] = round((res_growth.trend_lower.iloc[-1] * (1 - MAPE) / res_growth.shift_trend_lower.iloc[-1] - 1) * 100, 4)

        growth_with_validation['Верхняя граница прогноза'] = round(res_growth.trend_upper.iloc[-1], 4)
        growth_with_validation['Верхняя граница прогноза (с поправкой на ошибку модели)'] = round(res_growth.trend_upper.iloc[-1] * (1 + MAPE), 4)
        growth_with_validation['Верхняя граница прогноза в %'] = round(res_growth.diff_trend_upper.iloc[-1] * 100, 4)
        growth_with_validation['Верхняя граница прогноза (с поправкой на ошибку модели) в %'] = round((res_growth.trend_upper.iloc[-1] * (1 + MAPE) / res_growth.shift_trend_upper.iloc[-1] - 1) * 100, 4)

        display(pd.DataFrame.from_dict(growth_with_validation, orient = 'index').T)
        
    # ============ Auto ============
    def to_vanga(self):
        self.draw_timeseries()
        self.search_params()
        self.init_and_predict_prophet()
        self.draw_trend()
        self.draw_table_of_future_trend()