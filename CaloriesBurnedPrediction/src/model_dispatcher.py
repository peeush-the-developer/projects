''' Dispatch models that we want to run training for.

Dictionary <Model_Name, Model>
'''

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

models = {
    'lr': LinearRegression(),
    'dt': DecisionTreeRegressor(random_state=42)
}