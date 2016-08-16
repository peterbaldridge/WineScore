import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sknn.mlp import Regressor, Layer

# Import and Process Data
data = pd.read_csv("data/winequality-white.csv",sep=';')
y = data['quality']
data.drop(['quality'],axis=1,inplace=True)
X = data

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Create Model
nn = Regressor(
    layers=[
        Layer("Sigmoid", units=25),
        Layer("Sigmoid", units=25),
        Layer("Linear")],
    n_iter=5)

# Fit model
nn.fit(X[:4000],y[:4000])

# Mean Squared Error
mse = sklearn.metrics.mean_squared_error(y[4000:],nn.predict(X[4000:]))
print("MSE: " + str(mse))

# Mean error
predict = nn.predict(X[4000:])
predict = [i[0] for i in predict]
actual = y[4000:]
mean_error = sum(abs(predict-actual))/len(actual)
print("Mean Error: " + str(mean_error))

# Accuracy
acc_threshold = 1.0
accurate = abs(actual-predict) > acc_threshold
accuracy = 1 - sum(accurate) / len(actual)
print("Accuracy: " + str(accuracy))
