import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Import and Process Data
data = pd.read_csv("data/winequality-white.csv",sep=';')
y = data['quality']
data.drop(['quality'],axis=1,inplace=True)
X = data

# Normalize Data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Create Model
sv = svm.SVR(C=0.08,)
# Fit model
sv.fit(X[:4000],y[:4000])

# Calculate MSE
mse = sklearn.metrics.mean_squared_error(y[4000:],sv.predict(X[4000:]))
print("MSE: " + str(mse))

# Mean error
predict = sv.predict(X[4000:])
actual = y[4000:]
mean_error = sum(abs(predict-actual))/len(actual)
print("Mean Error: " + str(mean_error))

# Accuracy
acc_threshold = 1.0
accurate = abs(actual-predict) > acc_threshold
accuracy = 1 - sum(accurate) / len(actual)
print("Accuracy: " + str(accuracy))