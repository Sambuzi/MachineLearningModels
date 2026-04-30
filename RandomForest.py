import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.tree import plot_tree
from XGBoost import create_dataset
from sklearn.model_selection import cross_val_score

#We use the same create_dataset function as in XGBoost.py to create the lagged features and target variable for our time series data. 
#The main function reads the data, prepares the training and testing sets, fits the Random Forest model, makes rolling predictions, evaluates the model using MAE, and visualizes the results. Additionally, it computes and prints statistics about the trees in the random forest and plots the first tree for visualization.
def main(csv_path='M3C_monthly.CSV', lookback=12, nfore=12):
	df = pd.read_csv(
		csv_path,
		sep=';',
		decimal=',',
		engine='python',
		on_bad_lines='skip',
		encoding='latin1',
	)
	# select the same series as forecast_mlp: row 505, columns from index 6 onwards
	rawdata = df.iloc[505, 6:].values.astype(float)
	ds = pd.Series(rawdata)
	ds.index = pd.RangeIndex(start=0, stop=len(ds))
	vals = ds.values.astype(float)

	X, y = create_dataset(vals, lookback)

	x_train = X[:-nfore]
	x_test = X[-nfore:]
	y_train = y[:-nfore]
	y_test = y[-nfore:]

	# Optuna tuning: evaluate candidates with 3-fold CV on the training set
	def objective(trial):
		params = {
			"n_estimators": trial.suggest_int("n_estimators", 50, 1000),
			"max_depth": trial.suggest_int("max_depth", 2, 30),
			"max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.2, 0.5, 0.8]),
			"min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
			"min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
			"bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
		}

		model = RandomForestRegressor(**params, random_state=1, n_jobs=-1)
		scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
		return -np.mean(scores)

	study = optuna.create_study(direction="minimize")
	study.optimize(objective, n_trials=40)

	print("Best MAE (CV):", study.best_value)
	print("Best params:")
	for k, v in study.best_params.items():
		print(f"  {k}: {v}")

	# create RFmodel with best params and fit on full training set
	RFmodel = RandomForestRegressor(**study.best_params, random_state=1, n_jobs=-1)
	RFmodel.fit(x_train, y_train)

	# rolling nfore-step prediction using RFmodel
	xinput = x_train[-1].astype(float).copy()
	yfore = []
	for i in range(nfore):
		pred = RFmodel.predict(xinput.reshape(1, -1))[0]
		yfore.append(pred)
		xinput = np.roll(xinput, -1)
		xinput[-1] = pred

	mae = mean_absolute_error(y_test, yfore)
	print("MAE={:.6f}".format(mae))

	plt.plot(vals, label="Actual series")
	plt.plot(range(len(vals) - nfore, len(vals)), yfore, "-o", label=f"{nfore}-forecast")
	plt.legend()
	plt.show()

	# Stats about the trees in the random forest,in this part we costruct the trees and then we compute rhe avarage number of nodes.
	n_nodes = []
	max_depths = []
	for ind_tree in RFmodel.estimators_:
		n_nodes.append(ind_tree.tree_.node_count)
		max_depths.append(ind_tree.tree_.max_depth)
	print(f'Average number of nodes {int(np.mean(n_nodes))}')
	print(f'Average maximum depth {int(np.mean(max_depths))}')

	# plot first tree (index 0)
	fig = plt.figure(figsize=(15, 10))
	feature_names = [f"lag_{i}" for i in range(lookback)]
	plot_tree(
		RFmodel.estimators_[0],
		max_depth=2,
		feature_names=feature_names,
		filled=True,
		impurity=True,
		rounded=True,
	)
	plt.show()
    
	# Recursive Feature Elimination 
	featNames = [f"lag_{i}" for i in range(lookback)]
	rfe = RFE(RFmodel, n_features_to_select=min(4, X.shape[1]))
	fit = rfe.fit(X, y) #using fit
	names = featNames
	predictors = []
	for i in range(len(fit.support_)):
		if fit.support_[i]:
			predictors.append(names[i])
	print("Columns with predictive power:", predictors)


if __name__ == '__main__':
	main()