"""
Модуль параллельной обработки данных.
"""

import argparse
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import functions as f
import pickle


def load_data(file_path):
    """
    Загружает объект данных из бинарного файла.
    :param file_path: путь к файлу.
    :return: загруженный объект данных.
    """
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
    return data


def get_map10_by_days_rates(precisions: pd.Series, data_path: Path) -> pd.Series:
    """
    Рассчитывает точность предсказаний по метрике MAP@10, полученных при фильтрации только по глубине
    на основе информации о количестве дней до последней транзакции при разных значениях коэффициента фильтрации.
    :param precisions: Pandas Series, индексом которого является список значений коэффициента фильтрации,
    а значениями np.nan
    :param data_path: путь к папке с данными.
    :return: Pandas Series со значениями метрики ``MAP@10``
    """

    prior_transactions = load_data(data_path / 'prior_transactions.dmp')
    last_products = load_data(data_path / 'last_products.dmp')

    for days_rate in precisions.index:
        map10 = f.get_prediction_precision(
            true=last_products,
            prediction=f.get_prediction(
                f.get_ratings(
                    f.get_weights(prior_transactions, days_rate=days_rate))),
            k=10
        )
        precisions.at[days_rate] = map10

    return precisions


def get_map10_by_cart_rates(precisions: pd.DataFrame, data_path: Path, days_rate: float):
    """
    Рассчитывает точность предсказаний по метрике MAP@10, полученных при фильтрации по глубине
    на основе информации о количестве дней до последней транзакции и фильтрации по номеру добавления продукта в корзину
    при разных значениях коэффициента фильтрации.
    :param precisions: Pandas Series, индексом которого является список значений коэффициента фильтрации,
    а значениями np.nan
    :param data_path: путь к папке с данными.
    :param days_rate: коэффициент фильтрации по времени.
    :return: Pandas Series со значениями метрики ``MAP@10``.
    """

    prior_transactions = load_data(data_path / 'prior_transactions.dmp')
    last_products = load_data(data_path / 'last_products.dmp')

    for cart_rate in precisions.index:
        map10 = f.get_prediction_precision(
            true=last_products,
            prediction=f.get_prediction(
                f.get_ratings(
                    f.get_weights(
                        prior_transactions, days_rate=days_rate, cart_rate=cart_rate))),
            k=10
        )
        precisions.at[cart_rate] = map10

    return precisions


def get_map10_by_total_rates(precisions: pd.DataFrame, data_path: Path,
                             days_rate: float, cart_rate: float):
    """
    Рассчитывает точность предсказаний по метрике MAP@10, полученных при фильтрации по глубине
    на основе информации о количестве дней до последней транзакции и фильтрации по номеру добавления продукта в корзину
    при разных значениях коэффициента фильтрации по глобальному рейтингу.
    :param precisions: Pandas Series, индексом которого является список значений коэффициента фильтрации,
    а значениями np.nan
    :param data_path: путь к папке с данными.
    :param days_rate: коэффициент фильтрации по времени.
    :param cart_rate: коэффициент фильтрации по номеру добавления товара в корзину.
    :return: Pandas Series со значениями метрики ``MAP@10``
    """

    prior_transactions = load_data(data_path / 'prior_transactions.dmp')
    last_products = load_data(data_path / 'last_products.dmp')

    weights = f.get_weights(prior_transactions, days_rate=days_rate, cart_rate=cart_rate)
    ratings = f.get_ratings(weights).rename(columns={'rating': 'user_rating'})
    total_ratings = f.get_total_ratings(weights).rename(columns={'rating': 'total_rating'})
    ratings = ratings.merge(total_ratings, on='product_id', how='left')

    for rate in precisions.index:
        ratings['rating'] = ratings['user_rating'] * np.exp(ratings['total_rating'] * rate)
        map10 = f.get_prediction_precision(
            true=last_products,
            prediction=f.get_prediction(ratings),
            k=10
        )
        precisions.at[rate] = map10

    return precisions


if __name__ == '__main__':

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", help="Number of parallel workers.")
    parser.add_argument("--data_path", help="Path to the data folder.")
    parser.add_argument("--start", help="Start value of var range.")
    parser.add_argument("--stop", help="Stop value of var range.")
    parser.add_argument("--num", help="Spacing between values in var range.")
    parser.add_argument("--func", help="Name of the calling function.")
    parser.add_argument("--days_rate", help="Rate of 'days_before_last_order' filtration.")
    parser.add_argument("--cart_rate", help="Rate of 'add_to_cart_order' filtration.")

    args = parser.parse_args()

    WORKERS = int(args.workers)
    DATA_PATH = Path(args.data_path)
    var_range = np.linspace(float(args.start), float(args.stop), int(args.num))
    func = locals()[args.func]

    precisions = pd.DataFrame(
        columns=['var', 'precision', 'worker'],
        dtype=int
    )

    precisions['var'] = var_range
    precisions['worker'] = precisions.index % WORKERS
    precisions.set_index('var', inplace=True)

    with Pool(WORKERS) as pool:
        if func == get_map10_by_days_rates:
            precisions.index.name = 'days_rate'
            process_results = [pool.apply_async(func, (data, DATA_PATH))
                               for _, data in precisions.groupby('worker')['precision']]
        elif func == get_map10_by_cart_rates:
            days_rate = float(args.days_rate)
            precisions.index.name = 'cart_rate'
            process_results = [pool.apply_async(func, (data, DATA_PATH, days_rate))
                               for _, data in precisions.groupby('worker')['precision']]
        elif func == get_map10_by_total_rates:
            days_rate = float(args.days_rate)
            cart_rate = float(args.cart_rate)
            precisions.index.name = 'total_rate'
            process_results = [pool.apply_async(func, (data, DATA_PATH, days_rate, cart_rate))
                               for _, data in precisions.groupby('worker')['precision']]

        result = pd.concat([process_result.get() for process_result in process_results]).sort_index()

    with open(DATA_PATH / 'precisions.dmp', 'wb') as fp:
        pickle.dump(result, fp)
