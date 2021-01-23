from sklearn.tree import DecisionTreeClassifier

__RANDOM_STATE__ = 42
MODELS = {
    'dt_g': DecisionTreeClassifier(criterion='gini', random_state=__RANDOM_STATE__),
    'dt_e': DecisionTreeClassifier(criterion='entropy', random_state=__RANDOM_STATE__)
}