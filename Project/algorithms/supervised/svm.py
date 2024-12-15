from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def svm(df):
    model = SVC(kernel='linear')
    model.fit(df.X_train, df.y_train)
    y_hat = model.predict(df.X_test)

    accuracy = accuracy_score(df.y_test, y_hat)

    return {
        'algorithm': 'svm',
        'type': 'supervised', 
        'accuracy': accuracy,
    }
