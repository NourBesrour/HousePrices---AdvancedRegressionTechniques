import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xg

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

object_columns_train = train_data.select_dtypes(include=['object']).columns.tolist()
object_columns_test = test_data.select_dtypes(include=['object']).columns.tolist()

label_encoder = LabelEncoder()
for column in object_columns_train:
    train_data[column] = label_encoder.fit_transform(train_data[column])

for column in object_columns_test:
    test_data[column] = label_encoder.fit_transform(test_data[column])

train_data_cl = train_data.query("GarageYrBlt.notnull()")

ss = StandardScaler()
x, y = train_data_cl.drop(columns='SalePrice', axis=1), train_data_cl['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

xgb_model = xg.XGBRegressor(learning_rate=0.2, max_depth=3, min_child_weight=2, n_estimators=200)
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("XGBoost RMSE:", round(rmse, 2))

# Corrected line: Pass the test_data to ss.transform()
y_pred_test = xgb_model.predict(ss.transform(test_data.values))

# Ensure that 'test_data' has the same length as 'y_pred_test'
test_data['ID'] = range(1461, 1461 + len(test_data))

results_test = pd.DataFrame({
    'ID': test_data['ID'],
    'SalePrice': y_pred_test
})

# Save the DataFrame with results to a CSV file
results_test.to_csv('result_predictions.csv', index=False)
print("Result predictions saved to 'result_predictions.csv'")
