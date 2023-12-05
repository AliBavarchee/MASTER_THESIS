
class DeepRandomForestRegressor(nn.Module):
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super(DeepRandomForestRegressor, self).__init__()
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=0,  # Set a fixed random state for reproducibility
        )
        #: number of particle types
        #self.n_class = n_class
        n_class = 6

        #: number of detectors

        #self.n_detector = n_detector
        n_detector = 6

        self.fcs = nn.ModuleList([nn.Linear(self.n_detector, 1, bias=False) for _ in range(self.n_class)])
        #self.fc1 = nn.Linear(6, 36)  # Input features = 6, Hidden layer with 36 neurons
        #self.fc2 = nn.Linear(36, 1)  # Output layer with 1 neuron

​
    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        n = self.n_detector
        outs = [self.fcs[i](x[:, i * n: (i + 1) * n]) for i in range(self.n_class)]
        out = torch.cat(outs, dim=1)
        x = F.softmax(out, dim=1)
        return x
​
    def fit(self, X, y):
        # Train the RandomForestRegressor on the original data
        self.rf.fit(X, y)
​
        # Check if the RandomForestRegressor has the 'feature_importances_' attribute
        if hasattr(self.rf, 'feature_importances_'):
            feature_importances = self.rf.feature_importances_
​
            # Normalize feature importances so that they sum to 1
            normalized_importances = feature_importances / np.sum(feature_importances)
​
            # Replace the feature importances in the DeepRandomForestRegressor
            with torch.no_grad():
                self.fc1.weight.data = torch.tensor(normalized_importances[:, np.newaxis], dtype=torch.float32)
                self.fc1.bias.data = torch.zeros(self.fc1.bias.data.shape, dtype=torch.float32)
​
    def predict(self, X):
        # Use the RandomForestRegressor to get predictions
        return self.rf.predict(X)
​
    def score(self, X, y):
        # Use the mean squared error to evaluate the performance
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        return mse

# Define the RF_NET class that contains an RFRegressor 
#+ fit(train) + predict + score(print out mse) and finall gets 6 x 6 RF_weight MAtrix===>
class RF_NET:
    def __init__(self, **params):
        self.drf = DeepRandomForestRegressor(**params)
​
    def train(self, X_train, y_train):
        self.drf.fit(X_train, y_train)
​
    def predict(self, X_test):
        return self.drf.predict(X_test)
​
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse
​
    def get_weights(self, to_numpy=True):
        
        """Returns the feature importances as a six-by-six array or tensor.
​
        Args:
            to_numpy (bool, optional): Whether to return the weights as a numpy
                array (True) or torch tensor (False). Defaults to True.
​
        Returns:
            np.array or torch.Tensor: The six-by-six matrix containing the
                feature importances.
                
        """
        if self.drf.feature_importances_ is None:
            return None
        
​
        feature_importances = self.drf.feature_importances_
        weights = np.zeros((6, 6))
        idx = 0
        for i in range(6):
            
            for j in range(i, 6):
                weights[i][j] = feature_importances[idx]
                weights[j][i] = feature_importances[idx]
                idx += 1
        if to_numpy:
            return weights
        else:
            return torch.tensor(weights)