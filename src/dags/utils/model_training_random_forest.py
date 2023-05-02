import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

import config
from helper import get_logger

# init logging
logger = get_logger()

# load dataset
data = pd.read_parquet(config.ENG_DATASET_PATH)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
logger.info("Finished loading the dataset")

# Select features and target
features = ['vol_moving_avg', 'adj_close_rolling_med']
target = 'Volume'

# build x, y
X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=1)

del data

# Create a RandomForestRegressor model
model = RandomForestRegressor(
    n_estimators=50, 
    max_depth=10, 
    random_state=1, 
    n_jobs=4
)
logger.info("Random Forest model is built")

# Train the model
model.fit(X_train, y_train)
logger.info("Random Forest model training is completed")

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

logger.info("----------------      Random Forest      --------------------")
logger.info(f"Random Forest MAE: {mae}")
logger.info(f"Random Forest MSE: {mse}")
logger.info(f"Random Forest EVS: {evs}")
logger.info(f"Random Forest R^2: {r2}")
logger.info("-------------------------------------------------------------")

# Save the model to disk
# a better apprach would be MLflow
joblib.dump(model, config.RF_MODEL_PATH)