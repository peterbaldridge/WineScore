import pandas as pd
import sklearn
from sklearn import ensemble

# Import and Process Data
data = pd.read_csv("data/winequality-white.csv",sep=';')
y = data['quality']
data.drop(['quality'],axis=1,inplace=True)
X = data

# Fit model
rf = ensemble.RandomForestRegressor(n_estimators=100,max_features=4,n_jobs=-1,min_samples_split=20,warm_start=False,max_depth=15)
rf.fit(X[:4000],y[:4000])

# Mean Squared Error
mse = sklearn.metrics.mean_squared_error(y[4000:],rf.predict(X[4000:]))
print("MSE: " + str(mse))

# Mean error
predict = rf.predict(X[4000:])
actual = y[4000:]
mean_error = sum(abs(predict-actual))/len(actual)
print("Mean Error: " + str(mean_error))

# Accuracy
acc_threshold = 1.0
accurate = abs(actual-predict) > acc_threshold
accuracy = 1 - sum(accurate) / len(actual)
print("Accuracy: " + str(accuracy))
