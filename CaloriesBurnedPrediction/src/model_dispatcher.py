''' Dispatch models that we want to run training for.

Dictionary <Model_Name, Model>
'''

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = {
    'lr': LinearRegression(),
    'dt': DecisionTreeRegressor(random_state=42),
    'rfe': RandomForestRegressor(n_estimators=30, random_state=42)
}