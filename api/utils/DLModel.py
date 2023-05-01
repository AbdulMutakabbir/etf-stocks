import torch 
class DLModel(torch.nn.Module):
    
    def __init__(self, stats:dict):
        super(DLModel, self).__init__()
        self.hidden_layer_size = 128
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