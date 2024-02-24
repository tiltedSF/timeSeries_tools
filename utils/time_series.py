import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid

from prophet import Prophet
from utils import rrcf
from utils import shingle

class time_series:
    def __init__(self, df, interval=90, test_size=30, interval_width=0.95, loss='MAPE'):
        """
        :param df: <pd.DataFrame> with columns = | ds <datetime> | y <int, float> | - our TS
        :param interval: <int> - the interval for how far into the future to predict
        :param test_size: <int> - size of the test data for validation
        :param interval_width: <float> - confidence interval width
        :param loss: <string> - the error that will be used for the grid_search, one of the presented
            ['MSE', 'MAE', 'MAPE']
        """
        self.data = df.copy()
        self.data['ds'] = pd.to_datetime(self.data['ds'])
        self.interval = interval
        self.test_size = test_size
        self.interval_width = interval_width
        self.loss = loss

        self.data_train = self.data.iloc[: -test_size]
        self.data_test = self.data.iloc[-test_size:]

        self.parametres_dict = {'changepoint_prior_scale': 0.05,
                                'seasonality_prior_scale': 10.,
                                'holidays_prior_scale': 10.,
                                'seasonality_mode': 'additive',
                                'changepoint_range': 0.8}
        self.losses_with_parameters_table = None

    # ============ Setters ============
    def set_inteval(self, interval):
        """
        set interval for the future prediction
        :param interval: <int> - the interval for how far into the future to predict
        """
        self.interval = interval

    def set_test_size(self, test_size):
        """
        set test of the test data for validation
        :param test_size: <int> - size of the test data for validation
        """
        self.test_size = test_size
        self.data_train = self.data.iloc[:-test_size]
        self.data_test = self.data.iloc[-test_size:]

    def set_interval_width(self, interval_width):
        """
        set confidence interval width
        :param interval_width: <float> - confidence interval width
        """
        self.interval_width = interval_width

    def set_loss(self, loss):
        """
        set the error that will be used for the grid_search
        :param loss: <string> - the error that will be used for the grid_search, one of the presented
            ['MSE', 'MAE', 'MAPE']
        """
        self.loss = loss

    def set_custom_test_size(self, date_test_start):
        """
        set date from which the test sample will be started
        :param date_test_start: <string> - the date from which the test sample will be started
        """
        date_test_start = pd.Timestamp(date_test_start)
        self.data_train = self.data[(self.data.ds < date_test_start)]
        self.data_test = self.data[(self.data.ds >= date_test_start)]
        self.test_size = len(self.data_test)

    def set_model_parameters(self, dict_of_params):
        """
        set parameters for the model_prophet
        :param dict_of_params: <dict> - dict {'changepoint_prior_scale': <float>,
                                              'seasonality_prior_scale': <float>,
                                              'holidays_prior_scale': <float>,
                                              'seasonality_mode': <string> ['additive', 'multiplicative', 'auto'],
                                              'changepoint_range': <float>}
        """
        for param in dict_of_params:
            self.parametres_dict[param] = dict_of_params[param]

    # ============ Getters ============

    # ============ Model ============
    def search_params(self, changepoint_prior_scale=[0.01, 0.05, 0.1],
                      seasonality_prior_scale=[0.01, 0.1, 1, 10],
                      holidays_prior_scale=[0.01, 0.1, 1, 10],
                      seasonality_mode=['additive', 'multiplicative'],
                      changepoint_range=[0.8, 0.9, 0.95]):
        """
        grid_search for the parameters for model_prophet
<string> ['additive', 'multiplicative'],
        :param changepoint_prior_scale: <list><float>,
        :param seasonality_prior_scale: <list><float>
        :param holidays_prior_scale: <list><float>
        :param seasonality_mode: <list><string> ['additive', 'multiplicative', 'auto']
        :param changepoint_range: <list><float>
        :print: errors value at the validation, params after grid_search
        """

        params_grid = {'changepoint_prior_scale': changepoint_prior_scale,
                       'seasonality_prior_scale': seasonality_prior_scale,
                       'holidays_prior_scale': holidays_prior_scale,
                       'seasonality_mode': seasonality_mode,
                       'changepoint_range': changepoint_range}
        grid = ParameterGrid(params_grid)

        model_parameters = pd.DataFrame(columns=['MSE', 'MAE', 'MAPE',
                                                 'changepoint_prior_scale', 'seasonality_prior_scale',
                                                 'holidays_prior_scale', 'seasonality_mode', 'changepoint_range'])
        for p in tqdm(grid):
            random.seed(0)

            train_model = Prophet(interval_width=self.interval_width,
                                  changepoint_prior_scale=p['changepoint_prior_scale'],
                                  seasonality_prior_scale=p['seasonality_prior_scale'],
                                  holidays_prior_scale=p['holidays_prior_scale'],
                                  seasonality_mode=p['seasonality_mode'],
                                  changepoint_range=p['changepoint_range']
                                  )
            train_model.add_country_holidays(country_name='Russia')
            train_model.fit(self.data_train)

            train_forecast = train_model.make_future_dataframe(periods=self.test_size, freq='D', include_history=False)
            train_forecast = train_model.predict(train_forecast)

            test = train_forecast[['ds', 'yhat']]
            Actual = self.data_test

            MSE = mean_squared_error(Actual['y'], test['yhat'])
            MAE = mean_absolute_error(Actual['y'], test['yhat'])
            MAPE = mean_absolute_percentage_error(Actual['y'], abs(test['yhat']))

            model_parameters = pd.concat([model_parameters, pd.DataFrame({'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE,
                                                                          'changepoint_prior_scale': p[
                                                                              'changepoint_prior_scale'],
                                                                          'seasonality_prior_scale': p[
                                                                              'seasonality_prior_scale'],
                                                                          'holidays_prior_scale': p[
                                                                              'holidays_prior_scale'],
                                                                          'seasonality_mode': p['seasonality_mode'],
                                                                          'changepoint_range': p['changepoint_range']}
                                                                         , index=[0])]).reset_index(drop=True)

        self.losses_with_parameters_table = model_parameters.sort_values(by=[self.loss]).reset_index(drop=True)

        self.parametres_dict = {
            'changepoint_prior_scale': self.losses_with_parameters_table.changepoint_prior_scale[0],
            'seasonality_prior_scale': self.losses_with_parameters_table.seasonality_prior_scale[0],
            'holidays_prior_scale': self.losses_with_parameters_table.holidays_prior_scale[0],
            'seasonality_mode': self.losses_with_parameters_table.seasonality_mode[0],
            'changepoint_range': self.losses_with_parameters_table.changepoint_range[0]
        }

        print('Ошибки на валидации:')
        print(f"MSE = {self.losses_with_parameters_table['MSE'][0]}")
        print(f"MAE = {self.losses_with_parameters_table['MAE'][0]}")
        print(f"MAPE = {self.losses_with_parameters_table['MAPE'][0]}")
        print('Текущие параметры после gridSearch:', end='\n\n')
        for param in self.parametres_dict:
            print(f'{param} = {self.parametres_dict[param]}')

    def init_and_predict_prophet(self):
        """
        init model_prophet and predict future
        """
        self.prophet_model = Prophet(interval_width=self.interval_width,
                                     changepoint_prior_scale=self.parametres_dict['changepoint_prior_scale'],
                                     seasonality_prior_scale=self.parametres_dict['seasonality_prior_scale'],
                                     holidays_prior_scale=self.parametres_dict['holidays_prior_scale'],
                                     seasonality_mode=self.parametres_dict['seasonality_mode'],
                                     changepoint_range=self.parametres_dict['changepoint_range']
                                     )
        self.prophet_model.add_country_holidays(country_name='Russia')

        # Обучаем
        self.prophet_model.fit(self.data)
        # Выдаем предикт
        future_df = self.prophet_model.make_future_dataframe(periods=self.interval, freq='D', include_history=True)
        self.forecast = self.prophet_model.predict(future_df)
        # Выдать ошибки на валидации
        if self.losses_with_parameters_table is None:
            test = self.forecast[['ds', 'yhat']].iloc[-(self.test_size + self.interval):-self.interval]
            MSE = mean_squared_error(self.data_test['y'], test['yhat'])
            MAE = mean_absolute_error(self.data_test['y'], test['yhat'])
            MAPE = mean_absolute_percentage_error(self.data_test['y'], abs(test['yhat']))

            self.losses_with_parameters_table = pd.DataFrame({'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE,
                                                         'changepoint_prior_scale': self.parametres_dict[
                                                             'changepoint_prior_scale'],
                                                         'seasonality_prior_scale': self.parametres_dict[
                                                             'seasonality_prior_scale'],
                                                         'holidays_prior_scale': self.parametres_dict[
                                                             'holidays_prior_scale'],
                                                         'seasonality_mode': self.parametres_dict['seasonality_mode'],
                                                         'changepoint_range': self.parametres_dict['changepoint_range']}
                                                        , index=[0])
        print('Ошибки на валидации:')
        print(f"MSE = {self.losses_with_parameters_table['MSE'][0]}")
        print(f"MAE = {self.losses_with_parameters_table['MAE'][0]}")
        print(f"MAPE = {self.losses_with_parameters_table['MAPE'][0]}")

    # ============ Anomalys ============
    def anomalys_dates(self, num_trees = 40, shingle_size = 7, tree_size = 256):
        """
        Основано на RRCT - двоичном дереве поиска.
        (глубина конкретной точки в дереве эквивалентна ее битовой глубине)
        Под выбросом понимается точка, которая значительно увеличивает сложность модели
        при включении ее в дерево.
        Количественно выражая, можно определить смещение, вызванное точкой x,
        как ожидаемое изменение битовой глубины всех листьев в дереве RRCT, если точка x будет удалена.
        Disp(x, Z) = SUM(Pr[T](f(y, Z, T) - f(y, Z - x, T))), где
            x - удаляемая точка,
            Z - множество точек,
            T - дерево RRCT,
            f(y,Z,T) - глубина точки y в дереве T, определенном на множестве точек Z.
        CoDisp расширяет понятие смещения за счет учета дубликатов и близких дубликатов,
        которые могут маскировать присутствие выбросов.
        CoDisp(x, Z, |S|) = E[max_(S⊆Z,T)(1/|C|)SUM(f(y, S, T) - f(y, S - C, T''))], где
            x - это точка, которую нужно удалить,
            C - набор "сговорщиков", которые будут удалены вместе с x
        CoDisp пытается найти наименьшее подмножество точек C⊇x,
        которое максимизирует общее перемещение, если одновременно удалить все точки в C.
        ---------------------------------------------------------------
        Reference:
        Guha, S., Mishra, N., Roy, G., Schrijvers, O., 2016a.
        Robust random cut forest based anomaly detection on streams.
        In: International conference on machine learning. pp. 2712–2721

        :param num_trees: <int>
        :param shingle_size: <int>
        :param tree_size: <int>
        :return: pd.DataFrame with columns = | ds | y | CoDisp |
        :print: dates with CoDisp >= .99 perc
        """
        tmp_data = self.data.copy()
        data_for_anomalys = tmp_data['y'].to_numpy()

        # Create a forest of empty trees
        forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            forest.append(tree)

        # Use the "shingle" generator to create rolling window
        points = shingle.shingle(data_for_anomalys, size=shingle_size)

        # Create a dict to store anomaly score of each point
        avg_codisp = {}

        # For each shingle
        for index, point in enumerate(points):
            # For each tree in the forest
            for tree in forest:
                # If tree is above permitted size
                if len(tree.leaves) > tree_size:
                    # Drop the oldest point (FIFO)
                    tree.forget_point(index - tree_size)
                # Insert the new point into the tree
                tree.insert_point(point, index=index)
                # Compute codisp on the new point
                new_codisp = tree.codisp(index)
                # And take the average over all trees
                if not index in avg_codisp:
                    avg_codisp[index] = 0
                avg_codisp[index] += new_codisp / num_trees

        dates = tmp_data.iloc[0:len(tmp_data) - shingle_size + 1]['ds']
        avg_codisp = pd.DataFrame({'ds': dates, 'CoDisp': avg_codisp.values()})
        res = tmp_data.merge(avg_codisp, on='ds')

        res['ds'] = pd.to_datetime(res['ds'])

        fig, ax1 = plt.subplots(figsize=(20, 10))
        blue, red = 'tab:blue', 'tab:red'

        ax1.set_ylabel('Реальная метрика', color=blue, size=14)
        ax1.plot(res['ds'].to_numpy(), res['y'].to_numpy(), color=blue)
        ax1.tick_params(axis='y', labelcolor=blue, labelsize=12)
        ax1.set_ylim(min(res['y'].to_numpy()) * 0.9, max(res['y'].to_numpy()) * 1.1)

        ax1.tick_params(axis="x", labelrotation=55)
        ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m.%Y"))

        ax2 = ax1.twinx()

        ax2.set_ylabel('CoDisp', color=red, size=14)
        ax2.plot(res['ds'].to_numpy(), res['CoDisp'].to_numpy(), color=red)
        ax2.tick_params(axis='y', labelcolor=red, labelsize=12)
        ax2.grid('off')
        ax2.set_ylim(min(res['CoDisp'].to_numpy()) * 0.9, max(res['CoDisp'].to_numpy()) * 1.1)
        plt.show()

        print("Даты, которые выглядят очень аномально\n(Обрезаны по .99 квантилю)")
        anomalys_dates = res[res.CoDisp >= np.quantile(res.CoDisp, 0.99)]['ds']
        for date in anomalys_dates: print(date.date())

        return res

    # ============ Draw graphics ============
    def draw_timeseries(self):
        """
        :print: graph of self.data
        """
        plt.title('Реальная метрика')
        plt.plot(self.data['ds'].to_numpy(), self.data['y'].to_numpy(), label='real')
        plt.show()

    def draw_test_and_train_dataset(self):
        """
        :print: graph of self.data with train and test data
        """
        plt.title(f'График test and train')
        plt.plot(self.data_train['ds'].to_numpy(), self.data_train['y'].to_numpy(), label='train')
        plt.plot(self.data_test['ds'].to_numpy(), self.data_test['y'].to_numpy(), label='test')
        plt.legend()
        plt.show()

    def draw_forecast(self):
        """
        :print: graph with the forecast after prediction
        """
        plt.title(f'График для будущего')
        plt.plot(self.data['ds'].to_numpy(), self.data['y'].to_numpy(), label='real')
        plt.plot(self.forecast['ds'].to_numpy(), self.forecast['yhat'].to_numpy(), label='forecast')
        plt.fill_between(self.forecast['ds'].to_numpy(),
                         self.forecast['yhat_lower'].to_numpy(), self.forecast['yhat_upper'].to_numpy(),
                         alpha=0.3, color='grey', label=f'дов. интервал {self.interval_width}')
        plt.legend()
        plt.show()

    def draw_trend(self):
        """
        :print: graph with trend after prediction
        """
        plt.title('График для тренда на будущее')
        plt.plot(self.forecast['ds'].to_numpy(), self.forecast['trend_upper'].to_numpy(), label='upper')
        plt.plot(self.forecast['ds'].to_numpy(), self.forecast['trend_lower'].to_numpy(), label='lower')
        plt.fill_between(self.forecast['ds'].to_numpy(),
                         self.forecast['trend_lower'].to_numpy(), self.forecast['trend_upper'].to_numpy(),
                         alpha=0.5, color='grey', label=f'дов. интервал {self.interval_width}')
        plt.legend()
        plt.show()

    def plot_components(self):
        """
        plot componets of the model
        :print: graphs with componets of the model
        """
        self.prophet_model.plot_components(self.forecast);

    def draw_table_of_future_quater_trend(self, future_quater=None):
        """
        :param future_quater: the next quarter, which we've been forecasting
        :print: table with expected growth adjusted for model error on validation
        """

        MAPE = self.losses_with_parameters_table['MAPE'][0]

        if future_quater is None:
            future_quater = str(pd.to_datetime(self.data.ds.max()).year) + 'Q' + str(
                pd.to_datetime(self.data.ds.max()).quarter)

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
        growth_with_validation['Нижняя граница прогноза (с поправкой на ошибку модели)'] = round(
            res_growth.trend_lower.iloc[-1] * (1 - MAPE), 4)
        growth_with_validation['Нижняя граница прогноза в %'] = round(res_growth.diff_trend_lower.iloc[-1] * 100, 4)
        growth_with_validation['Нижняя граница прогноза (с поправкой на ошибку модели) в %'] = round(
            (res_growth.trend_lower.iloc[-1] * (1 - MAPE) / res_growth.shift_trend_lower.iloc[-1] - 1) * 100, 4)

        growth_with_validation['Верхняя граница прогноза'] = round(res_growth.trend_upper.iloc[-1], 4)
        growth_with_validation['Верхняя граница прогноза (с поправкой на ошибку модели)'] = round(
            res_growth.trend_upper.iloc[-1] * (1 + MAPE), 4)
        growth_with_validation['Верхняя граница прогноза в %'] = round(res_growth.diff_trend_upper.iloc[-1] * 100, 4)
        growth_with_validation['Верхняя граница прогноза (с поправкой на ошибку модели) в %'] = round(
            (res_growth.trend_upper.iloc[-1] * (1 + MAPE) / res_growth.shift_trend_upper.iloc[-1] - 1) * 100, 4)

        display(pd.DataFrame.from_dict(growth_with_validation, orient='index').T)

    # ============ Auto ============
    def to_vanga(self):
        """
        if you're lazy and want to push one button and get something
        """
        self.draw_timeseries()
        self.search_params()
        self.init_and_predict_prophet()
        self.draw_forecast()
        self.draw_trend()
        self.draw_table_of_future_quater_trend()
