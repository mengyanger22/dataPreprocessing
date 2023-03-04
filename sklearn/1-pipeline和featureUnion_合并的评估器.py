'''
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA


estimators = [("reduce_dim", PCA()), ("clf", SVC())]
pipe = Pipeline(estimators)
# print(pipe)
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
tmp = make_pipeline(Binarizer(), MultinomialNB())
# print(tmp)
# print(pipe.steps[0])
# print(pipe[0])
# print(pipe["reduce_dim"])
tmp = pipe.named_steps.reduce_dim is pipe["reduce_dim"]
# print(tmp)
# print(pipe[:1])
# print(pipe[-1:])
tmp = pipe.set_params(clf__C=10)
# print(tmp)

from sklearn.model_selection import GridSearchCV
param_grid = dict(reduce_dim__n_components=[2,5,10], clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)
# print(grid_search)

from sklearn.linear_model import LogisticRegression
param_grad = dict(reduce_dim=["passthrough", PCA(5), PCA(10)], 
                  clf=[SVC(), LogisticRegression()],
                  clf__C=[0.1, 10, 100]
                  )
grid_search = GridSearchCV(pipe, param_grid=param_grad)
# print(grid_search)
# print(pipe[0])

# 实例1：Pipeline ANOVA SVM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_features=20,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

anova_filter = SelectKBest(f_classif, k=3)
clf = LinearSVC()
anova_svm = make_pipeline(anova_filter, clf)
tmp = anova_svm.fit(X_train, y_train)
# print(tmp)

from sklearn.metrics import classification_report
y_pred = anova_svm.predict(X_test)
tmp = classification_report(y_test, y_pred)
# print(tmp)
tmp = anova_svm[-1].coef_
# print(tmp)
tmp = anova_svm[:-1].inverse_transform(anova_svm[-1].coef_)
print(tmp)

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

estimators = [("reduce_dim", PCA()), ("clf", SVC())]
cachedir = mkdtemp()
pipe = Pipeline(estimators, memory=cachedir)
# print(pipe)
rmtree(cachedir)

from sklearn.datasets import load_digits
digits = load_digits()
pca1 = PCA()
svm1 = SVC(gamma="scale")
pipe = Pipeline([("reduce_dim", pca1), ("clf", svm1)])
tmp = pipe.fit(digits.data, digits.target)
# print(tmp)
# print(pca1.components_)

cachedir = mkdtemp()
pca2 = PCA()
svm2 = SVC(gamma="scale")
cached_pipe = Pipeline([("reduce_dim", pca2), ["clf", svm2]], memory=cachedir)
tmp = cached_pipe.fit(digits.data, digits.target)
# print(tmp)
# print(pca2.components_)
print(cached_pipe.named_steps["reduce_dim"].components_)
rmtree(cachedir)
'''
'''
import numpy as np
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

load_boston = fetch_california_housing
boston = load_boston()
X, y = boston.data, boston.target

transformer = QuantileTransformer(output_distribution="normal")
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor, transformer=transformer)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
tmp = regr.fit(X_train, y_train)
# print(tmp)
# print("R2 score: {0:.2f}".format(regr.score(X_test, y_test)))
# print("raw R2 score:{0:.2f}".format(LinearRegression().fit(X_train, y_train).score(X_test, y_test)))

def func(x):
    return np.log(x)

def inverse_func(x):
    return np.exp(x)

regr = TransformedTargetRegressor(regressor=regressor,
                                  func=func,
                                  inverse_func=inverse_func
                                  )
tmp = regr.fit(X_train, y_train)
# print(tmp)
# print("R2 score: {0:.2f}".format(regr.score(X_test, y_test)))

def inverse_func(x):
    return x

regr = TransformedTargetRegressor(
    regressor=regressor,
    func=func,
    inverse_func=inverse_func,
    check_inverse=False,
)

regr.fit(X_train, y_train)
# print("R2 score: ", regr.score(X_test, y_test))

'''
'''
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

estimators = [("linear_pca", PCA()), ("kernel_pca", KernelPCA())]
combined = FeatureUnion(estimators)
# print(combined)
tmp = combined.set_params(kernel_pca="drop")
# print(tmp)
'''

import pandas as pd
X = pd.DataFrame({
    "city" : ["London", "London", "Paris", "Sallisaw"],
    "title" : ["His Last Bow", "How watson Learned the Trick", "A Moveable Feast", "The Grapes of Wrath"],
    "expert_rating" : [5, 3, 4, 5],
    "user_rating" : [4, 5, 4, 3],
})

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder


column_trans = ColumnTransformer(
    [
    ("city_category", CountVectorizer(analyzer=lambda x: [x]), "city"),
    ("title_bow", CountVectorizer(), "title")
    ],
    remainder="drop"
)

tmp = column_trans.fit(X)
# print(tmp)
tmp = column_trans.get_feature_names()
# print(tmp) 
tmp = column_trans.transform(X)
# print(tmp)

column_trans = ColumnTransformer(
    [
    ("city_category", OneHotEncoder(dtype="int"), ["city"]),
    ("title_bow", CountVectorizer(), "title"),
     ],
     remainder="passthrough"
)

tmp = column_trans.fit_transform(X)
# print(tmp)

from sklearn.preprocessing import MinMaxScaler
column_trans = ColumnTransformer(
    [("city_category", OneHotEncoder(), ["city"]),
     ("title_bow", CountVectorizer(), "title"),
     ],
     remainder=MinMaxScaler()
)

tmp = column_trans.fit_transform(X)[:, -2:]
# print(tmp)

from sklearn.compose import make_column_transformer

column_trans = make_column_transformer(
    (OneHotEncoder(), ["city"]),
    (CountVectorizer(), "title"),
    remainder=MinMaxScaler()
)
print(column_trans)
