from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def decision_tree(df):
    clf = DecisionTreeClassifier()
    clf.fit(df.X_train, df.y_train)
    y_hat = clf.predict(df.X_test)
    accuracy = accuracy_score(df.y_test, y_hat)
        
    return {
        'algorithm': 'decision tree',
        'type': 'supervised', 
        'accuracy': accuracy,
    }