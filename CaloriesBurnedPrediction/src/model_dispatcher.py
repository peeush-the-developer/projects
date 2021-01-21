''' Dispatch models that we want to run training for.

Dictionary <Model_Name, Model>
'''

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

__RANDOM_STATE__ = 42

models = {
    'lr': LinearRegression(),
    'dt': DecisionTreeRegressor(random_state=__RANDOM_STATE__),
    'rf': RandomForestRegressor(n_estimators=30, random_state=__RANDOM_STATE__),
    'lasso': Lasso(random_state=__RANDOM_STATE__)
}