import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import pandas as pd
from interpretability.explain import explain_model

def train():
    data = load_boston()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train.values, dtype=torch.float32))
        loss = criterion(outputs.squeeze(), torch.tensor(y_train.values, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    preds = model(torch.tensor(X_test.values, dtype=torch.float32)).detach().numpy().squeeze()
    mse = mean_squared_error(y_test, preds)
    print(f"Test MSE: {mse}")
    explain_model(model, X_test)
