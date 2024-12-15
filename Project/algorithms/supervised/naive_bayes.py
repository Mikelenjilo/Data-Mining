from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def naive_bayes(df):
    gnb = GaussianNB()
    gnb.fit(df.X_train, df.y_train)
    y_hat = gnb.predict(df.X_test)

    accuracy = accuracy_score(df.y_test, y_hat)
        
    return {
        'algorithm': 'naive bayes',
        'type': 'supervised', 
        'accuracy': accuracy,
    }