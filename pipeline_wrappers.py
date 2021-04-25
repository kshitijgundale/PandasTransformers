import pandas as pd
import numpy as np

class PandasTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns
        self.feature_names = None

    def fit(self, X, y=None, **fit_params):
        if self.columns == "same":
          self.feature_names = X.columns
        else:
          self.feature_names = self.columns
        return self.transformer.fit(X)

    def transform(self, X, y=None, **transform_params):
        arr = self.transformer.transform(X)
        if isinstance(arr, np.ndarray):
          return pd.DataFrame(data=arr, columns=self.feature_names)
        elif isinstance(arr, pd.DataFrame):
          return arr
        else:
          return pd.DataFrame(data=arr.toarray(), columns=self.feature_names)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

    def get_feature_names(self):
        return list(self.columns)

class PandasFeatureUnion(BaseEstimator, TransformerMixin):

    def __init__(self, transformers, **kwargs):
        self.transformers = transformers
        self.col_names = []

    def fit(self, X, y=None, **fit_params):
        for name, transformer, cols in self.transformers:
          transformer.fit(X[cols])
        return self
    
    def transform(self, X, y=None, **transform_params):
        transformer_dataframes = []
        for name, transformer, cols in self.transformers:
          df = transformer.transform(X[cols])
          assert isinstance(df, pd.DataFrame)
          self.col_names.append(df.columns)
          transformer_dataframes.append(df)
        return pd.concat(transformer_dataframes, axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

class CombineFeatures(TransformerMixin):

  def __init__(self, sep, name):
      self.sep = sep
      self.name = name

  def transform(self, X, **transform_params):
      df = X.apply(lambda x : " ".join(x), axis=1)
      return df

  def fit(self, X, y=None, **fit_params):
      return self
    
  def fit_transform(self, X, y=None, **fit_params):
      self.fit(X)
      return self.transform(X)

  def get_feature_names(self):
      return self.name