''' Dispatch models that we want to run training for.

Dictionary <Model_Name, Model>
'''

from sklearn.linear_model import LinearRegression

models = {
    'lr': LinearRegression()
}