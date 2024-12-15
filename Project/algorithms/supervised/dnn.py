from keras import Sequential
from sklearn.metrics import accuracy_score
import keras
import numpy as np

def dnn(df, nb_hidden_layers, nb_nodes, target_column):
    max_accuracy = 0
    max_nb_hidden = 0
    max_nb_nodes = 0

    for nb_hidden in nb_hidden_layers:
        for nb_node in nb_nodes:
            accuracy = 0
            nb_attributs = df.X_test.shape[1]
            
            model = Sequential()
            
            model.add(keras.layers.Dense(nb_attributs, activation='relu'))
            for _ in range(nb_hidden):
                    model.add(keras.layers.Dense(nb_node, activation='relu'))
                    
            nb_classes = df.df[target_column].nunique()
            
            if nb_classes == 2:
                model.add(keras.layers.Dense(1, activation='sigmoid'))
                
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                model.fit(df.X_train, df.y_train, epochs=50)
                y_hat = model.predict(df.X_test)
                y_hat = (y_hat >= 0.5).astype(int)
                
                accuracy = accuracy_score(df.y_test, y_hat)

            else:
                model.add(keras.layers.Dense(nb_classes, activation='softmax'))
                
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                model.fit(df.X_train, df.y_train, epochs=50)
                y_hat = model.predict(df.X_test)
                y_hat = np.argmax(y_hat, axis=1)
                
                accuracy = accuracy_score(df.y_test, y_hat)
                
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_nb_hidden = nb_hidden
                max_nb_nodes = nb_node
        
    return {
        'algorithm': 'dnn',
        'type': 'supervised', 
        'accuracy': max_accuracy,
        'nb hidden layers': max_nb_hidden,
        'nb nodes per hidden layer': max_nb_nodes,
    }