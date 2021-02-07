import plotly.express as px
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.feature_column as fc
from tensorflow.keras.datasets import boston_housing

################ Simple linear regression ################################
# Generate random data. Add noise. 
np.random.seed(0)
area = 2.5*np.random.randn(100) + 25
price = 25*area + 5 + np.random.randint(20, 50, size=len(area))

# Combine data in df
data = np.array([area, price])
data = pd.DataFrame(data = data.T, columns=['area', 'price'])

plt.scatter(data.area, data.price)
plt.show()

# Apply the regression equations, for each coefficient
W = sum(price*(area-np.mean(area))) / sum((area-np.mean(area))**2)
b = np.mean(price) - W*np.mean(area)
print("The regression coefficients are", W, b)

# Make predictions using the computed values 
y_pred = W * area + b

# Plot predictions 
plt.plot(area, y_pred, color='r', label='Predicted price')
plt.scatter(data.area, data.price, c='k', label='Training data')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()

################ Multiple linear regression ################################
# Using TF estimators for two independent variables (two features), predict home price
numeric_column = fc.numeric_column
categorical_column_with_vocabulary_list = fc.categorical_column_with_vocabulary_list

# Define feature columns used to train regressor 
featcols = [
    tf.feature_column.numeric_column('area'),
    tf.feature_column.categorical_column_with_vocabulary_list("type", [
        'bungalow', 'apartment'])
]

# Func to provide input for training
def train_input_fn():
    features = {'area': [1000, 2000, 4000, 1000, 2000, 4000],
                'type': ['bungalow', 'bungalow', 'house',
                 'apartment', 'apartment', 'apartment']}
    labels = [500, 1000, 1500, 700, 1300, 1900]
    return features, labels

# Use premade linear regressor estimator and fit on training
model = tf.estimator.LinearRegressor(featcols)
model.train(train_input_fn, steps=200)

# Test on prediction
def predict_input_fn():
    features = {"area": [1500, 1800],
                "type": ["house", "apt"]}
    return features
 
predictions = model.predict(predict_input_fn)

################ Multiple linear regression on real data ################################
# Download data set
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

features = ['CRIM', 'ZN',
            'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x_train_df = pd.DataFrame(x_train, columns=features)
x_test_df = pd.DataFrame(x_test, columns=features)
y_train_df = pd.DataFrame(y_train, columns=['MEDV'])
y_test_df = pd.DataFrame(y_test, columns=['MEDV'])
x_train_df.head()

features_checkcc = ['ZN', 'INDUS', 'NOX', 'RM', 'AGE',
            'DIS', 'PTRATIO', 'B', 'LSTAT']

feature_columns = []
for feature_name in features_checkcc:
    feature_columns.append(fc.numeric_column(feature_name,
                                             dtype=tf.float32))

# Create the input function for the estimator
# The function returns the tf.Data.Dataset object with a tuple: features and labels in batches

def estimator_input_fn(df_data, df_label, epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(df_data), df_label))
        if shuffle:
            ds = ds.shuffle(100)
        ds = ds.batch(batch_size).repeat(epochs)
        return ds
    return input_function


train_input_fn = estimator_input_fn(x_train_df, y_train_df)
val_input_fn = estimator_input_fn(x_test_df, y_test_df, epochs=1,
                                  shuffle=False)

# Initiate linear regressor estimator
linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
linear_est.train(train_input_fn, steps=100)
result = linear_est.evaluate(val_input_fn)

for pred, exp in zip(result, y_test[:32]):
    print("Predicted Value: ", pred['predictions'][0], "Expected:", exp)
