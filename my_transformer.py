from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.impute import SimpleImputer


class MyTransformer:
    """
    A class for performing data transformation on a DataFrame using various techniques.

    Args:
        all_features (List[str]): A list of all feature names in the DataFrame.
        categorical_features (List[str]): A list of categorical feature names in the DataFrame.
        normal_transform_features (List[str]): A list of feature names to be normalized using MinMaxScaler.
        power_transform_features (List[str]): A list of feature names to be transformed using PowerTransformer.
        label_column (str, optional): The name of the label column. Defaults to None.
    """
    label_column: str | None
    all_features: list[str]
    skewed_features: list[str]
    normal_features: list[str]
    numerical_features: list[str]
    normal_feature_transformer = MinMaxScaler
    skewed_feature_transformer = PowerTransformer
    label_transformer = MinMaxScaler

    def __init__(self,
                 all_features: List[str],
                 categorical_features: List[str],
                 normal_transform_features: List[str],
                 power_transform_features: List[str],
                 numerical_features: List[str],
                 label_column: str = None
                 ):
        """
        Initialize MyTransformer with the specified configuration.
        """
        self.normal_feature_transformer = MinMaxScaler()
        self.skewed_feature_transformer = PowerTransformer()
        self.label_transformer = MinMaxScaler()
        self.all_features = all_features
        self.categorical_features = categorical_features
        self.normal_features = normal_transform_features
        self.skewed_features = power_transform_features
        self.label_column = label_column
        self.numerical_features = numerical_features
        self._is_fit = False
        self.imputer = SimpleImputer(strategy='mean')

    def features_fit_transform(self, df: pd.DataFrame):
        """
        Fits and transforms the input DataFrame using the specified transformations.

        Args:
            df (pd.DataFrame): The input DataFrame to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        self._is_fit = True

        # fit & transform features
        o_df = df.copy()[self.all_features]
        o_df[self.normal_features] = self.normal_feature_transformer.fit_transform(df[self.normal_features])
        o_df[self.skewed_features] = self.skewed_feature_transformer.fit_transform(df[self.skewed_features])
        o_df = pd.get_dummies(df, columns=self.categorical_features)
        o_df = o_df.astype(float)

        # for each numerical features, add a new column to indicate whether it is missing
        for feature in self.numerical_features:
            o_df[feature + '_missing'] = df[feature].isna().astype(float)

        # replace all Nan to 0
        o_df = self.imputer.fit_transform(o_df)

        return o_df

    def label_fit_transform(self, df: pd.DataFrame):
        # fit & transform label
        o_df = df.copy()
        return self.label_transformer.fit_transform(o_df[[self.label_column]])

    def features_transform(self, df: pd.DataFrame):
        """
        Transforms the input DataFrame using the previously fitted transformations.
        :param df: The input DataFrame to be transformed.
        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        assert self._is_fit, "Must fit first"

        # fit & transform features
        o_df = df.copy()[self.all_features]
        o_df[self.normal_features] = self.normal_feature_transformer.transform(df[self.normal_features])
        o_df[self.skewed_features] = self.skewed_feature_transformer.transform(df[self.skewed_features])
        o_df = pd.get_dummies(df, columns=self.categorical_features)
        o_df = o_df.astype(float)

        # for each numerical features, add a new column to indicate whether it is missing
        for feature in self.numerical_features:
            o_df[feature + '_missing'] = df[feature].isna().astype(float)

        # use sklearn mean imputation to replace all Nan
        o_df = self.imputer.transform(o_df)

        return o_df

    def label_transform(self, df: pd.DataFrame):
        # fit & transform label
        o_df = df.copy()
        return self.label_transformer.transform(o_df[[self.label_column]])

    def inverse_label_transform(self, arr: np.array):
        """
        Inverse transform the label column.
        :param arr:
        :return:
        """
        """arr is a 1d array"""
        assert self._is_fit, "Must fit first"
        return self.label_transformer.inverse_transform([arr])[0]
