# coding:utf-8

import os
import gc
import time
import datetime
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor

from sklearn.externals import joblib
np.random.seed(7)


class GbcTry(object):
    def __init__(self, input_path, output_path):
        self.__input_path, self.__output_path = input_path, output_path

        # read data
        self.__train, self.__test = [None for _ in range(2)]
        self.__train_feature, self.__test_feature = [None for _ in range(2)]
        self.__train_label, self.__test_label = [None for _ in range(2)]
        self.__sample_submission = None

        # prepare data

        # transform data
        self.__numeric_columns, self.__categorical_columns = [None for _ in range(2)]
        self.__encoder = None

        # model fit
        self.__xgb_bo = None
        self.__xgb_params = None
        self.__xgb = None

        # model predict

        # write data

    def read_data(self):
        self.__train = pd.read_csv(os.path.join(self.__input_path, "train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__input_path, "test.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__input_path, "sample_submission.csv"))

    def prepare_data(self):
        # train
        self.__train = self.__train.drop(["Id"], axis=1)
        self.__train_label = self.__train["SalePrice"].copy()
        self.__train_feature = self.__train.drop(["SalePrice"], axis=1).copy()
        del self.__train
        gc.collect()

        # test
        self.__test_feature = self.__test[self.__train_feature.columns.tolist()].copy()
        del self.__test
        gc.collect()

    def transform_data(self):
        self.__train_feature["MSZoning"] = self.__train_feature["MSZoning"].replace("C (all)", "C")
        self.__test_feature["MSZoning"] = self.__test_feature["MSZoning"].replace("C (all)", "C")

        # MSSubClass "_" MSZoning
        self.__train_feature["MSSubClass_MSZoning"] = (
            self.__train_feature["MSSubClass"].astype(str) + "_" + self.__train_feature["MSZoning"].astype(str))
        self.__train_feature = self.__train_feature.drop(["MSSubClass", "MSZoning"], axis=1)
        self.__test_feature["MSSubClass_MSZoning"] = (
            self.__test_feature["MSSubClass"].astype(str) + "_" + self.__test_feature["MSZoning"].astype(str))
        self.__test_feature = self.__test_feature.drop(["MSSubClass", "MSZoning"], axis=1)

        # Street "_" Alley
        self.__train_feature["Street_Alley"] = (
            self.__train_feature["Street"].astype(str) + "_" + self.__train_feature["Alley"].astype(str))
        self.__train_feature = self.__train_feature.drop(["Street", "Alley"], axis=1)
        self.__test_feature["Street_Alley"] = (
            self.__test_feature["Street"].astype(str) + "_" + self.__test_feature["Alley"].astype(str))
        self.__test_feature = self.__test_feature.drop(["Street", "Alley"], axis=1)

        # LandContour  "_" LandSlope
        self.__train_feature["LandContour_LandSlope"] = (
            self.__train_feature["LandContour"].astype(str) + "_" + self.__train_feature["LandSlope"].astype(str))
        self.__train_feature = self.__train_feature.drop(["LandContour", "LandSlope"], axis=1)
        self.__test_feature["LandContour_LandSlope"] = (
            self.__test_feature["LandContour"].astype(str) + "_" + self.__test_feature["LandSlope"].astype(str))
        self.__test_feature = self.__test_feature.drop(["LandContour", "LandSlope"], axis=1)

        # LotShape "_" LotConfig
        self.__train_feature["LotShape_LotConfig"] = (
            self.__train_feature["LotShape"].astype(str) + "_" + self.__train_feature["LotConfig"].astype(str))
        self.__train_feature = self.__train_feature.drop(["LotShape", "LotConfig"], axis=1)
        self.__test_feature["LotShape_LotConfig"] = (
            self.__test_feature["LotShape"].astype(str) + "_" + self.__test_feature["LotConfig"].astype(str))
        self.__test_feature = self.__test_feature.drop(["LotShape", "LotConfig"], axis=1)

        # Neighborhood  "_" Condition1 "_" Condition2
        self.__train_feature["Neighborhood_Condition1_Condition2"] = (
            self.__train_feature["Neighborhood"].astype(str) + "_" +
            self.__train_feature["Condition1"].astype(str) + "_" +
            self.__train_feature["Condition2"].astype(str))
        self.__train_feature = self.__train_feature.drop(["Neighborhood", "Condition1", "Condition2"], axis=1)
        self.__test_feature["Neighborhood_Condition1_Condition2"] = (
            self.__test_feature["Neighborhood"].astype(str) + "_" +
            self.__test_feature["Condition1"].astype(str) + "_" +
            self.__test_feature["Condition2"].astype(str))
        self.__test_feature = self.__test_feature.drop(["Neighborhood", "Condition1", "Condition2"], axis=1)

        # OverallQual "_" OverallCond
        self.__train_feature["OverallQual_OverallCond"] = (
            self.__train_feature["OverallQual"].astype(str) + "_" + self.__train_feature["OverallCond"].astype(str))
        self.__train_feature = self.__train_feature.drop(["OverallQual", "OverallCond"], axis=1)
        self.__test_feature["OverallQual_OverallCond"] = (
            self.__test_feature["OverallQual"].astype(str) + "_" + self.__test_feature["OverallCond"].astype(str))
        self.__test_feature = self.__test_feature.drop(["OverallQual", "OverallCond"], axis=1)

        # SUB(YearRemodAdd, YearBuilt)
        self.__train_feature["SUB(YearRemodAdd, YearBuilt)"] = (
            self.__train_feature["YearRemodAdd"] - self.__train_feature["YearBuilt"])
        self.__test_feature["SUB(YearRemodAdd, YearBuilt)"] = (
            self.__test_feature["YearRemodAdd"] - self.__test_feature["YearBuilt"])

        # SUB(NOW, YearBuilt)
        self.__train_feature["SUB(NOW, YearBuilt)"] = (
            time.localtime().tm_year - self.__train_feature["YearBuilt"])  # time.localtime() 得到当前年份
        self.__test_feature["SUB(NOW, YearBuilt)"] = (
            time.localtime().tm_year - self.__test_feature["YearBuilt"])

        # SUB(NOW, YearRemodAdd)
        self.__train_feature["SUB(NOW, YearRemodAdd)"] = (
            time.localtime().tm_year - self.__train_feature["YearRemodAdd"])  # time.localtime() 得到当前年份
        self.__test_feature["SUB(NOW, YearRemodAdd)"] = (
            time.localtime().tm_year - self.__test_feature["YearRemodAdd"])

        self.__train_feature = self.__train_feature.drop(["YearBuilt", "YearRemodAdd"], axis=1)
        self.__test_feature = self.__test_feature.drop(["YearBuilt", "YearRemodAdd"], axis=1)

        # RoofStyle "_" RoofMatl
        self.__train_feature["RoofStyle_RoofMatl"] = (
            self.__train_feature["RoofStyle"].astype(str) + "_" + self.__train_feature["RoofMatl"].astype(str))
        self.__train_feature = self.__train_feature.drop(["RoofStyle", "RoofMatl"], axis=1)
        self.__test_feature["RoofStyle_RoofMatl"] = (
            self.__test_feature["RoofStyle"].astype(str) + "_" + self.__test_feature["RoofMatl"].astype(str))
        self.__test_feature = self.__test_feature.drop(["RoofStyle", "RoofMatl"], axis=1)

        # Exterior1st "_" Exterior2nd
        self.__train_feature["Exterior1st_Exterior2nd"] = (
            self.__train_feature["Exterior1st"].astype(str) + "_" + self.__train_feature["Exterior2nd"].astype(str))
        self.__train_feature = self.__train_feature.drop(["Exterior1st", "Exterior2nd"], axis=1)
        self.__test_feature["Exterior1st_Exterior2nd"] = (
            self.__test_feature["Exterior1st"].astype(str) + "_" + self.__test_feature["Exterior2nd"].astype(str))
        self.__test_feature = self.__test_feature.drop(["Exterior1st", "Exterior2nd"], axis=1)

        # ExterQual "_" ExterCond
        self.__train_feature["ExterQual_ExterCond"] = (
            self.__train_feature["ExterQual"].astype(str) + "_" + self.__train_feature["ExterCond"].astype(str))
        self.__train_feature = self.__train_feature.drop(["ExterQual", "ExterCond"], axis=1)
        self.__test_feature["ExterQual_ExterCond"] = (
            self.__test_feature["ExterQual"].astype(str) + "_" + self.__test_feature["ExterCond"].astype(str))
        self.__test_feature = self.__test_feature.drop(["ExterQual", "ExterCond"], axis=1)

        # BsmtQual "_" BsmtCond "_" BsmtExposure "_" BsmtFinType1 "_" BsmtFinType2
        self.__train_feature["BsmtQual_BsmtCond_BsmtExposure_BsmtFinType1_BsmtFinType2"] = (
            self.__train_feature["BsmtQual"].astype(str) + "_" +
            self.__train_feature["BsmtCond"].astype(str) + "_" +
            self.__train_feature["BsmtExposure"].astype(str) + "_" +
            self.__train_feature["BsmtFinType1"].astype(str) + "_" +
            self.__train_feature["BsmtFinType2"].astype(str))
        self.__train_feature = self.__train_feature.drop(["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"], axis=1)
        self.__test_feature["BsmtQual_BsmtCond_BsmtExposure_BsmtFinType1_BsmtFinType2"] = (
            self.__test_feature["BsmtQual"].astype(str) + "_" +
            self.__test_feature["BsmtCond"].astype(str) + "_" +
            self.__test_feature["BsmtExposure"].astype(str) + "_" +
            self.__test_feature["BsmtFinType1"].astype(str) + "_" +
            self.__test_feature["BsmtFinType2"].astype(str))
        self.__test_feature = self.__test_feature.drop(["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"], axis=1)

        # Heating "_" HeatingQC
        self.__train_feature["Heating_HeatingQC"] = (
            self.__train_feature["Heating"].astype(str) + "_" + self.__train_feature["HeatingQC"].astype(str))
        self.__train_feature = self.__train_feature.drop(["Heating", "HeatingQC"], axis=1)
        self.__test_feature["Heating_HeatingQC"] = (
            self.__test_feature["Heating"].astype(str) + "_" + self.__test_feature["HeatingQC"].astype(str))
        self.__test_feature = self.__test_feature.drop(["Heating", "HeatingQC"], axis=1)

        # KitchenAbvGr "_" KitchenQual
        self.__train_feature["KitchenAbvGr_KitchenQual"] = (
            self.__train_feature["KitchenAbvGr"].astype(str) + "_" + self.__train_feature["KitchenQual"].astype(str))
        self.__train_feature = self.__train_feature.drop(["KitchenAbvGr", "KitchenQual"], axis=1)
        self.__test_feature["KitchenAbvGr_KitchenQual"] = (
            self.__test_feature["KitchenAbvGr"].astype(str) + "_" + self.__test_feature["KitchenQual"].astype(str))
        self.__test_feature = self.__test_feature.drop(["KitchenAbvGr", "KitchenQual"], axis=1)

        # Fireplaces "_" FireplaceQu
        self.__train_feature["Fireplaces_FireplaceQu"] = (
            self.__train_feature["Fireplaces"].astype(str) + "_" + self.__train_feature["FireplaceQu"].astype(str))
        self.__train_feature = self.__train_feature.drop(["Fireplaces", "FireplaceQu"], axis=1)
        self.__test_feature["Fireplaces_FireplaceQu"] = (
            self.__test_feature["Fireplaces"].astype(str) + "_" + self.__test_feature["FireplaceQu"].astype(str))
        self.__test_feature = self.__test_feature.drop(["Fireplaces", "FireplaceQu"], axis=1)

        # GarageType "_" GarageYrBlt "_" GarageFinish "_" GarageCars "_" GarageArea "_" GarageQual "_" GarageCond
        self.__train_feature["GarageType_GarageYrBlt_GarageFinish_GarageCars_GarageArea_GarageQual_GarageCond"] = (
            self.__train_feature["GarageType"].astype(str) + "_" +
            self.__train_feature["GarageYrBlt"].astype(str) + "_" +
            self.__train_feature["GarageFinish"].astype(str) + "_" +
            self.__train_feature["GarageCars"].astype(str) + "_" +
            self.__train_feature["GarageArea"].astype(str) + "_" +
            self.__train_feature["GarageQual"].astype(str) + "_" +
            self.__train_feature["GarageCond"].astype(str))
        self.__train_feature = self.__train_feature.drop(["GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond"], axis=1)
        self.__test_feature["GarageType_GarageYrBlt_GarageFinish_GarageCars_GarageArea_GarageQual_GarageCond"] = (
            self.__test_feature["GarageType"].astype(str) + "_" +
            self.__test_feature["GarageYrBlt"].astype(str) + "_" +
            self.__test_feature["GarageFinish"].astype(str) + "_" +
            self.__test_feature["GarageCars"].astype(str) + "_" +
            self.__test_feature["GarageArea"].astype(str) + "_" +
            self.__test_feature["GarageQual"].astype(str) + "_" +
            self.__test_feature["GarageCond"].astype(str))
        self.__test_feature = self.__test_feature.drop(["GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond"], axis=1)

        # OpenPorchSF "_" EnclosedPorch "_" 3SsnPorch "_" ScreenPorch
        self.__train_feature["OpenPorchSF_EnclosedPorch_3SsnPorch_ScreenPorch"] = (
            self.__train_feature["OpenPorchSF"].astype(str) + "_" +
            self.__train_feature["EnclosedPorch"].astype(str) + "_" +
            self.__train_feature["3SsnPorch"].astype(str) + "_" +
            self.__train_feature["ScreenPorch"].astype(str))
        self.__train_feature = self.__train_feature.drop(["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"], axis=1)
        self.__test_feature["OpenPorchSF_EnclosedPorch_3SsnPorch_ScreenPorch"] = (
            self.__test_feature["OpenPorchSF"].astype(str) + "_" +
            self.__test_feature["EnclosedPorch"].astype(str) + "_" +
            self.__test_feature["3SsnPorch"].astype(str) + "_" +
            self.__test_feature["ScreenPorch"].astype(str))
        self.__test_feature = self.__test_feature.drop(["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"], axis=1)

        # YrSold "-" MoSold
        self.__train_feature["YrSold_MoSold"] = (
            self.__train_feature["YrSold"].astype(str) + "-" + self.__train_feature["MoSold"].astype(str) + "-" + "1")
        self.__train_feature = self.__train_feature.drop(["YrSold", "MoSold"], axis=1)
        self.__test_feature["YrSold_MoSold"] = (
            self.__test_feature["YrSold"].astype(str) + "-" + self.__test_feature["MoSold"].astype(str) + "-" + "1")
        self.__test_feature = self.__test_feature.drop(["YrSold", "MoSold"], axis=1)

        # DATEDIFF(NOW, YrSold_MoSold)
        self.__train_feature["DATEDIFF(NOW, YrSold_MoSold)"] = (
            datetime.datetime.now() - pd.to_datetime(self.__train_feature["YrSold_MoSold"])).apply(lambda x: x.days)
        self.__test_feature["DATEDIFF(NOW, YrSold_MoSold)"] = (
                datetime.datetime.now() - pd.to_datetime(self.__test_feature["YrSold_MoSold"])).apply(lambda x: x.days)
        self.__train_feature = self.__train_feature.drop(["YrSold_MoSold"], axis=1)
        self.__test_feature = self.__test_feature.drop(["YrSold_MoSold"], axis=1)

        # SaleType "_" SaleCondition
        self.__train_feature["SaleType_SaleCondition"] = (
            self.__train_feature["SaleType"].astype(str) + "_" + self.__train_feature["SaleCondition"].astype(str))
        self.__train_feature = self.__train_feature.drop(["SaleType", "SaleCondition"], axis=1)
        self.__test_feature["SaleType_SaleCondition"] = (
            self.__test_feature["SaleType"].astype(str) + "_" + self.__test_feature["SaleCondition"].astype(str))
        self.__test_feature = self.__test_feature.drop(["SaleType", "SaleCondition"], axis=1)

        # numeric feature ------------------------------------
        # 1stFlrSF 2ndFlrSF LowQualFinSF GrLivArea
        self.__train_feature["ADD(1stFlrSF, 2ndFlrSF)"] = self.__train_feature["1stFlrSF"] + self.__train_feature["2ndFlrSF"]
        self.__train_feature["DIVIDE(2ndFlrSF, 1stFlrSF)"] = self.__train_feature["2ndFlrSF"] / self.__train_feature["1stFlrSF"].apply(lambda x: np.nan if x == 0 else x)
        self.__train_feature["DIVIDE(1stFlrSF, LotArea)"] = self.__train_feature["1stFlrSF"] / self.__train_feature["LotArea"].apply(lambda x: np.nan if x == 0 else x)
        self.__train_feature["DIVIDE(2ndFlrSF, LotArea)"] = self.__train_feature["2ndFlrSF"] / self.__train_feature["LotArea"].apply(lambda x: np.nan if x == 0 else x)
        self.__train_feature["DIVIDE(ADD(1stFlrSF, 2ndFlrSF), LotArea)"] = self.__train_feature["ADD(1stFlrSF, 2ndFlrSF)"] / self.__train_feature["LotArea"].apply(lambda x: np.nan if x == 0 else x)

        self.__test_feature["ADD(1stFlrSF, 2ndFlrSF)"] = self.__test_feature["1stFlrSF"] + self.__test_feature["2ndFlrSF"]
        self.__test_feature["DIVIDE(2ndFlrSF, 1stFlrSF)"] = self.__test_feature["2ndFlrSF"] / self.__test_feature["1stFlrSF"].apply(lambda x: np.nan if x == 0 else x)
        self.__test_feature["DIVIDE(1stFlrSF, LotArea)"] = self.__test_feature["1stFlrSF"] / self.__test_feature["LotArea"].apply(lambda x: np.nan if x == 0 else x)
        self.__test_feature["DIVIDE(2ndFlrSF, LotArea)"] = self.__test_feature["2ndFlrSF"] / self.__test_feature["LotArea"].apply(lambda x: np.nan if x == 0 else x)
        self.__test_feature["DIVIDE(ADD(1stFlrSF, 2ndFlrSF), LotArea)"] = self.__test_feature["ADD(1stFlrSF, 2ndFlrSF)"] / self.__test_feature["LotArea"].apply(lambda x: np.nan if x == 0 else x)

        self.__categorical_columns = self.__train_feature.select_dtypes(include=object).columns.tolist()
        self.__numeric_columns = self.__train_feature.select_dtypes(exclude=object).columns.tolist()

        # encoder
        self.__encoder = TargetEncoder()
        self.__encoder.fit(self.__train_feature[self.__categorical_columns], self.__train_label)
        self.__train_feature[self.__categorical_columns] = (
            self.__encoder.transform(self.__train_feature[self.__categorical_columns]))
        self.__test_feature[self.__categorical_columns] = (
            self.__encoder.transform(self.__test_feature[self.__categorical_columns]))

        # self.__train_feature.to_csv("train_feature.csv", index=False)
        # self.__train_label.to_frame("SalePrice").to_csv("train_label.csv", index=False)
        # self.__test_feature.to_csv("test_feature.csv", index=False)

    def model_fit(self):
        def __cv(n_estimators, learning_rate, subsample, colsample_bytree):
            val = cross_val_score(
                XGBRegressor(
                    n_estimators=max(int(round(n_estimators)), 1),
                    learning_rate=max(min(learning_rate, 1.0), 0),
                    subsample=max(min(subsample, 1.0), 0),
                    colsample_bytree=max(min(colsample_bytree, 1.0), 0),
                    n_jobs=-1,
                    silent=True
                ),
                self.__train_feature,
                np.log1p(self.__train_label),
                scoring="neg_mean_squared_error",
                cv=KFold(n_splits=3, shuffle=True, random_state=7)
            ).mean()

            return val

        self.__xgb_params = {
            "n_estimators": (450, 750),
            "learning_rate": (0.001, 0.1),
            "subsample": (0.6, 1),
            "colsample_bytree": (0.6, 1)
        }

        self.__xgb_bo = BayesianOptimization(__cv, self.__xgb_params)
        self.__xgb_bo.maximize( init_points=5, n_iter=25, alpha=1e-4)

        self.__xgb = XGBRegressor(
            n_estimators=max(int(round(self.__xgb_bo.res["max"]["max_params"]["n_estimators"])), 1),
            learning_rate=max(min(self.__xgb_bo.res["max"]["max_params"]["learning_rate"], 1.0), 0),
            subsample=max(min(self.__xgb_bo.res["max"]["max_params"]["subsample"], 1.0), 0),
            colsample_bytree=max(min(self.__xgb_bo.res["max"]["max_params"]["colsample_bytree"], 1.0), 0)
        )
        self.__xgb.fit(self.__train_feature, np.log1p(self.__train_label))

    def model_predict(self):
        self.__sample_submission["SalePrice"] = (
                np.expm1(self.__xgb.predict(self.__test_feature)))

    def write_data(self):
        self.__sample_submission.to_csv(os.path.join(self.__output_path, "sample_submission.csv"), index=False)
        joblib.dump(self.__xgb, "GbcTry.z")


if __name__ == "__main__":
    gt = GbcTry(
        input_path="E:\\Kaggle\\House Price\\",
        output_path="E:\\Kaggle\\House Price\\"
    )
    gt.read_data()
    gt.prepare_data()
    gt.transform_data()
    gt.model_fit()
    gt.model_predict()
    gt.write_data()

