# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.datasets.california_housing import fetch_california_housing
np.random.seed(7)


class PartialDependencePlot(object):

    def __init__(self):
        self.__all = None
        self.__train_feature, self.__train_label = [None for _ in range(2)]
        self.__test_feature, self.__test_label = [None for _ in range(2)]

        self.__gbc = None

    def read_data(self):
        self.__all = fetch_california_housing()
        self.__train_feature, self.__test_feature, self.__train_label, self.__test_label = train_test_split(
            self.__all.data,
            self.__all.target,
            test_size=0.2,
            random_state=1
        )

    def model_fit(self):
        self.__gbc = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            loss="huber"
        )
        self.__gbc.fit(self.__train_feature, self.__train_label)

    def figure_plot(self):
        fig, _ = plot_partial_dependence(
            self.__gbc,
            self.__train_feature,
            features=self.__all.feature_names,
            feature_names=self.__all.feature_names,
            grid_resolution=100,
            n_cols=3
        )
        plt.show()


if __name__ == "__main__":
    pdp = PartialDependencePlot()
    pdp.read_data()
    pdp.model_fit()
    pdp.figure_plot()
