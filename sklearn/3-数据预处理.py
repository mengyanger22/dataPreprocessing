# from sklearn import preprocessing
# import numpy as np


# X_train = np.array(
#     [
#         [1., -1., 2.],
#         [2., 0., 0.],
#         [0., 1., -1.]
#     ]
# )
# X_scaled = preprocessing.scale(X_train)
# print(X_scaled)
# print(X_scaled.mean(axis=0))
# print(X_scaled.std(axis=0))

# # demonstration of the quantile transform
# import numpy as np
# from numpy import exp
# from numpy.random import randn
# from sklearn.preprocessing import QuantileTransformer
# import matplotlib.pyplot as plt


# # data = randn(2000)
# # data = exp(data)
# # # plt.hist(data, bins=25)
# # # plt.show()

# # data = data.reshape((len(data), 1))
# # quantile = QuantileTransformer(output_distribution="normal")
# # data_trans = quantile.fit_transform(data)
# # plt.hist(data_trans, bins=25)
# # plt.show()

# from pandas import read_csv
# from pandas.plotting import scatter_matrix
# from matplotlib import pyplot as plt


# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
# dataset = read_csv(url, header=None)
# print(dataset.shape)
# print(dataset.describe())
# dataset.hist()
# plt.show()

# from numpy import mean
# from numpy import std
# from pandas import read_csv
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt


# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
# dataset = read_csv(url, header=None)
# data = dataset.values
# X, y = data[:, :-1], data[:, -1]
# X = X.astype("float32")
# y = LabelEncoder().fit_transform(y.astype("str"))

# model = KNeighborsClassifier()
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1, error_score="raise")
# print("ACC: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))

# trans = QuantileTransformer(n_quantiles=100, output_distribution="normal")
# data = trans.fit_transform(data)

# from pandas import read_csv
# from pandas import DataFrame
# from pandas.plotting import scatter_matrix
# from sklearn.preprocessing import QuantileTransformer
# import matplotlib.pyplot as plt

# dataset = read_csv(url, header=None)
# data = dataset.values[:, :-1]
# trans = QuantileTransformer(n_quantiles=100, output_distribution="normal")
# data = trans.fit_transform(data)
# dataset = DataFrame(data)
# dataset.hist()
# plt.show()


# from numpy import mean
# from numpy import std
# from pandas import read_csv
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt


# dataset = read_csv(url, header=None)
# data = dataset.values
# X, y = data[:, :-1], data[:, -1]
# X = X.astype("float32")
# y = LabelEncoder().fit_transform(y.astype("str"))
# trans = QuantileTransformer(n_quantiles=100, output_distribution="normal")
# model = KNeighborsClassifier()
# pipeline = Pipeline(steps=[("t", trans), ("m", model)])
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(pipeline, X, y, scoring="accuracy", cv=cv, n_jobs=-1, error_score="raise")
# print("ACC: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))
# print(n_scores)

# from pandas import read_csv
# from pandas import DataFrame
# from pandas.plotting import scatter_matrix
# from sklearn.preprocessing import QuantileTransformer
# import matplotlib.pyplot as plt

# dataset = read_csv(url, header=None)
# data = dataset.values[:, :-1]
# trans = QuantileTransformer(n_quantiles=100, output_distribution="uniform")
# data = trans.fit_transform(data)
# dataset = DataFrame(data)
# dataset.hist()
# plt.show()

# from numpy import mean, std
# from pandas import read_csv
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder, QuantileTransformer
# from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt

# dataset = read_csv(url, header=None)
# data = dataset.values
# X, y = data[:, :-1], data[:, -1]
# X = X.astype("float32")
# y = LabelEncoder().fit_transform(y.astype("str"))
# trans = QuantileTransformer(n_quantiles=100, output_distribution="uniform")
# model = KNeighborsClassifier()
# pipeline = Pipeline(steps=[("t", trans), ("m", model)])
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(pipeline, X, y, scoring="accuracy", cv=cv, n_jobs=-1, error_score="raise")
# print("ACC %.3f (%.3f)" % (mean(n_scores), std(n_scores)))

# from numpy import std, mean
# from pandas import read_csv
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import QuantileTransformer, LabelEncoder
# from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt

# def get_dataset():
#     url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
#     dataset = read_csv(url, header=None)
#     data = dataset.values
#     X, y = data[:, :-1], data[:, -1]
#     X = X.astype("float32")
#     y = LabelEncoder().fit_transform(y.astype("str"))
#     return X, y

# def get_models():
#     models = dict()
#     for i in range(1, 100):
#         trans = QuantileTransformer(n_quantiles=i, output_distribution="normal")
#         model = KNeighborsClassifier()
#         models[str(i)] = Pipeline([("t", trans), ("m", model)])
#     return models

# def evaluate_model(model, X, y):
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=1, error_score="raise")
#     return scores

# X, y = get_dataset()
# models = get_models()
# results = list()
# for name, model in models.items():
#     scores = evaluate_model(model, X, y)
#     results.append(mean(scores))
#     print(">%s  %.3f (%.3f)" % (name, mean(scores), std(scores)))
# plt.plot(results)
# plt.show()

# from sklearn.preprocessing import QuantileTransformer


# import numpy as np
# from scipy.sparse import csr_matrix

# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])

# csr = csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
# # print(csr)

# indptr = np.array([0, 2, 5, 7])
# indices = np.array([1, 3, 0, 1, 3, 0, 2])
# data = np.array([1, 2, 1, 1, 2, 2, 5])
# csr = csr_matrix((data, indices, indptr))
# # print(csr)

# print(csr.indptr)
# print(csr.data)
# print(csr.data[csr.indptr[0] : csr.indptr[1]])


# from scipy.stats import norm

# tmp = norm.cdf([0, 1, 2])
# print(tmp)

# x = np.array([1, 2, 3])
# mask = ~np.isnan(x)
# print(mask)
# idx = mask - 1 < 2
# print(idx)
# from scipy import stats

# eps = 1e-7
# clip_min = stats.norm.ppf(eps - np.spacing(1))
# print(clip_min)

# import numpy as np
# import matplotlib as mpl
# from matplotlib import pyplot as plt
# from matplotlib import cm

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import minmax_scale
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import Normalizer
# from sklearn.preprocessing import QuantileTransformer
# from sklearn.preprocessing import PowerTransformer

# from sklearn.datasets import fetch_california_housing
# dataset = fetch_california_housing()
# X_full, y_full = dataset.data, dataset.target
# feature_names = dataset.feature_names

# feature_mapping = {
#     "MedInc" : "Median income in block",
#     "HousAge" : "Median house age in block",
#     "AveRooms" : "Average number of rooms",
#     "Population" : "Block population",
#     "AveOccup" : "Average house occupancy",
#     "Latitude" : "House block latitude",
#     "Longitude" : "House block longitude",
# }

# features = ["MedInc", "AveOccup"]
# features_idx = [feature_names.index(feature) for feature in features]
# X = X_full[:, features_idx]
# distributions = [
#     ("Unscalued data", X),
#     ("Data after standard scaling", StandardScaler().fit_transform(X)),
#     ("Data after min-max scaling", MinMaxScaler().fit_transform(X)),
#     ("Data after max-abs scaling", MaxAbsScaler().fit_transform(X)),
#     (
#         "Data after robust scaling",
#         RobustScaler(quantile_range=(25, 75)).fit_transform(X),
#     ),
#     (
#         "Data after power transform (Yeo-Johnson)",
#         PowerTransformer(method="yeo-johnson").fit_transform(X),
#     ),
#     (
#         "Data after power transform (Box-Cox)",
#         PowerTransformer(method="box-cox").fit_transform(X),
#     ),
#     (
#         "Data after quantile transformation (uniform pdf)",
#         QuantileTransformer(output_distribution="uniform").fit_transform(X),
#     ),
#     (
#         "Data after quantile transformation (gaussian pdf)",
#         QuantileTransformer(output_distribution="normal").fit_transform(X),
#     ),
#     ("Data after sample-wise L2 normalizing", Normalizer().fit_transform(X)),
# ]

# y = minmax_scale(y_full)
# cmap = getattr(cm, "plasma_r", cm.hot_r)

# def create_axes(title, figsize=(16, 6)):
#     fig = plt.figure(figsize=figsize)
#     fig.suptitle(title)

#     left, width = 0.1, 0.22
#     bottom, height = 0.1, 0.7
#     bottom_h = height + 0.15
#     left_h = left + width + 0.02

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom_h, width, 0.1]
#     rect_histy = [left_h, bottom, 0.05, height]

#     ax_scatter = plt.axes(rect_scatter)
#     ax_histx = plt.axes(rect_histx)
#     ax_histy = plt.axes(rect_histy)

#     left = width + left + 0.2
#     left_h = left + width + 0.02

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom_h, width, 0.1]
#     rect_histy = [left_h, bottom, 0.05, height]

#     ax_scatter_zoom = plt.axes(rect_scatter)
#     ax_histx_zoom = plt.axes(rect_histx)
#     ax_histy_zoom = plt.axes(rect_histy)

#     left, width = width + left + 0.13, 0.01
#     rect_colorbar = [left, bottom, width, height]
#     ax_colorbar = plt.axes(rect_colorbar)

#     return (
#         (ax_scatter, ax_histy, ax_histx),
#         (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
#         ax_colorbar,
#     )


# def plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
#     ax, hist_X1, hist_X0 = axes

#     ax.set_title(title)
#     ax.set_xlabel(x0_label)
#     ax.set_ylabel(x1_label)

#     colors = cmap(y)
#     ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     ax.spines["left"].set_position(("outward", 10))
#     ax.spines["bottom"].set_position(("outward", 10))

#     hist_X1.set_ylim(ax.get_ylim())
#     hist_X1.hist(
#         X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
#     )
#     hist_X1.axis("off")

#     hist_X0.set_xlim(ax.get_xlim())
#     hist_X0.hist(
#         X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
#     )
#     hist_X0.axis("off")


# def make_plot(item_idx):
#     title, X = distributions[item_idx]
#     ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
#     axarr = (ax_zoom_out, ax_zoom_in)
#     plot_distribution(
#         axarr[0],
#         X,
#         y,
#         hist_nbins=200,
#         x0_label=feature_mapping[features[0]],
#         x1_label=feature_mapping[features[1]],
#         title="full data"
#     )

#     zoom_in_percentitle_range = (0, 99)
#     cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentitle_range)
#     cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentitle_range)
#     non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
#         X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1
#     )
#     plot_distribution(
#         axarr[1],
#         X[non_outliers_mask],
#         y[non_outliers_mask],
#         hist_nbins=50,
#         x0_label=feature_mapping[features[0]],
#         x1_label=feature_mapping[features[1]],
#         title="Zoom-in",
#     )

#     norm = mpl.colors.Normalize(y_full.min(), y_full.max())
#     mpl.colorbar.ColorbarBase(
#         ax_colorbar,
#         cmap=cmap,
#         norm=norm,
#         orientation="vertical",
#         label="Color mapping for values of y",
#     )


# make_plot(1)
# plt.show()



# from sklearn import preprocessing
# import numpy as np

# X_train = np.array([
#     [1., -2., 2.],
#     [2., 0., 1.],
#     [0., 1., 3.],
# ])
# X_scale = preprocessing.scale(X_train)
# # print(X_scale)
# # print(X_scale.mean(axis=1))
# # print(X_scale.std(axis=0))

# # scaler = preprocessing.StandardScaler().fit(X_train)
# # print(scaler)
# # print(scaler.mean_)
# # print(scaler.scale_)
# # tmp = scaler.transform(X_train)
# # print(tmp)
# # X_test = [[-1., 1., 0.]]
# # tmp = scaler.transform(X_test)
# # print(tmp)
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(X_train)
# print(X_train_minmax)
# X_test = np.array([[-3., -1, 4.]])
# X_test_minmax = min_max_scaler.transform(X_test)
# # print(X_test_minmax)
# print(min_max_scaler.scale_)
# print(min_max_scaler.min_)

# from sklearn import preprocessing
# import numpy as np

# X_train = np.array([
#     [1., -1., 2.],
#     [2., 0., 0.],
#     [0., 1., -1.]
# ])

# max_abs_scaler = preprocessing.MaxAbsScaler()
# X_train_maxabs = max_abs_scaler.fit_transform(X_train)
# # print(X_train_maxabs)
# X_test = np.array([[-3., -1., 4.]])
# X_test_maxabs = max_abs_scaler.transform(X_test)
# # print(X_test_maxabs)

# print(max_abs_scaler.scale_)

# import numpy as np
# from sklearn import impute
# # imp = Imputer(missing_values="NaN", strategy=)

# imp = impute.SimpleImputer(missing_values=np.nan, strategy="mean")
# X_train = np.array([[1, 2], [np.nan, 3], [7, 6]])
# imp.fit(X_train)
# X_test = [[np.nan, 2], [6, np.nan], [7, 6]]
# # print(imp.transform(X_train))
# # print(imp.transform(X_test))

# import scipy.sparse as sp

# X = [[1, 2], [0, 3], [7, 6]]
# imp = impute.SimpleImputer(missing_values=0, strategy="mean")
# imp.fit(X)
# X_test = [[0, 2], [6, 0], [7, 6]]
# print(imp.transform(X_test))

# import pandas as pd
# import numpy as np
# from sklearn import impute

# df = pd.DataFrame([
#     ["a", "x"],
#     [np.nan, "y"],
#     ["a", np.nan],
#     ["b", "y"],
# ], dtype="category")

# imp = impute.SimpleImputer(strategy="most_frequent")
# print(imp.fit_transform(df))

