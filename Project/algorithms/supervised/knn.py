from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def knn(df, k=None, auto=False):
    if not auto:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(df.X_train, df.y_train)
        y_hat = model.predict(df.X_test)

        accuracy = accuracy_score(df.y_test, y_hat)
            
        return {
            'algorithm': 'knn',
            'type': 'supervised', 
            'k': k,
            'accuracy': accuracy,
        }
    
    else: 
        max_accuracy = 0
        max_k = 0

        for k in range(1, min(30, df.df.shape[1])):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(df.X_train, df.y_train)
            y_hat = model.predict(df.X_test)

            accuracy = accuracy_score(df.y_test, y_hat)
            
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_k = k
                
            
        return {
            'algorithm': 'knn',
            'type': 'supervised', 
            'k': max_k,
            'accuracy': max_accuracy,
        }

