import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split

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

# Creating a datasets
class DLDataset(Dataset):
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame)->None:
        # normalize the data  
        X = (X - X.mean())/X.std()
        y = (y - y.mean())/y.std()

        # saving stats
        self.X_mean = X.mean()
        self.y_mean = y.mean()
        self.X_std = X.std()
        self.y_std = y.std()

        # convert to torch
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32)
        self.length = len(self.y)
    
    def __len__(self)->int:
        return self.length
    
    def __getitem__(self, index) -> tuple:
        return self.X[index], self.y[index]
    
train_dataset = DLDataset(X=X_train, y=y_train)
dl_dataset_stats = {
    'X_mean': train_dataset.X_mean,
    'y_mean': train_dataset.y_mean,
    'X_std': train_dataset.X_std,
    'y_std': train_dataset.y_std,
}
joblib.dump(dl_dataset_stats, config.DL_DATASET_STATS_PATH)
test_dataset = DLDataset(X=X_test, y=y_test)

# Creating DataLoader
train_loader = DataLoader(train_dataset, batch_size=32768, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32768)

# Create a Deep Learning Model
class DLModel(torch.nn.Module):
    
    def __init__(self, stats:dict=dl_dataset_stats):
        super(DLModel, self).__init__()
        self.hidden_layer_size = 2048
        self.negative_slope = 0.01
        self.stats = stats

        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(in_features=2, out_features=self.hidden_layer_size))
        self.layers.append(torch.nn.LeakyReLU(negative_slope=self.negative_slope))
        # self.layers.append(torch.nn.Linear(in_features=(self.hidden_layer_size * 1), out_features=(self.hidden_layer_size * 2)))
        # self.layers.append(torch.nn.LeakyReLU(negative_slope=self.negative_slope))
        # self.layers.append(torch.nn.Linear(in_features=(self.hidden_layer_size * 2), out_features=(self.hidden_layer_size * 1)))
        # self.layers.append(torch.nn.LeakyReLU(negative_slope=self.negative_slope))
        self.layers.append(torch.nn.Linear(in_features=self.hidden_layer_size, out_features=1))
        self.layers.append(torch.nn.LeakyReLU(negative_slope=self.negative_slope))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def predict(self, vol_moving_avg:float, adj_close_rolling_med:float):
        self.eval()
        # standardizing the data
        vol_moving_avg = (vol_moving_avg - self.stats['X_mean']['vol_moving_avg'])/self.stats['X_std']['vol_moving_avg']
        adj_close_rolling_med = (adj_close_rolling_med - self.stats['X_mean']['adj_close_rolling_med'])/self.stats['X_std']['adj_close_rolling_med']
        # converting to tensor
        x = [vol_moving_avg, adj_close_rolling_med]
        x = torch.tensor(x, dtype=torch.float32)
        # make predictions
        y_pred = self.forward(x)
        # reverse the y standardization
        y_pred = (y_pred * self.stats['y_std']) + self.stats['y_mean']
        return y_pred

model = DLModel()
print(model)

# define your loss function
loss_fn = torch.nn.MSELoss()

# define your optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
logger.info(f"********************* Started training DL Model *********************************")
logger.info(model)
for epoch in range(5):
    total_loss = 0.0
    for X, y in tqdm(train_loader, unit="batch"):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_loss = total_loss/len(train_loader)
    logger.info(f"Epoch {epoch} loss: {epoch_loss}")

# Make predictions on test data
for X, y in test_loader:
    y_pred = model.forward(X).detach().numpy()
    logger.debug(f"Deep Learning Predictions: {y_pred}")
    break

# Calculate metrics
r2 = r2_score(y, y_pred)
evs = explained_variance_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
logger.info(f"Deep Learning Config: {config['deep_learning']}")
logger.info(f"Deep Learning MAE: {mae}")
logger.info(f"Deep Learning MSE: {mse}")
logger.info(f"Deep Learning EVS: {evs}")
logger.info(f"Deep Learning R^2: {r2}")
logger.info("-------------------------------------------------------------")

# Save the model to disk
# a better approach would be MLflow
torch.save(model.state_dict(), config.DL_MODEL_PATH)