from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from pandas import read_csv, concat
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
import numpy as np

class DF:
    def __init__(self, df=None, X_train=None, X_test=None, y_train=None, y_test=None, type=None):
        self.df = df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.type = type
        
    def reading_data(self, filepath, file_type='csv'):
        if file_type == 'csv':
            self.df = read_csv(filepath, na_values=['?', 'null', 'NaN'])
            self.type = 'csv'
        elif file_type == 'arff':
            data = loadarff(filepath)
            self.df = DataFrame(data[0])
            self.df = self.df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            self.type='arff'
        else:
            raise Exception('Type du fichier non supporté') 
        
    
    def preprocessing(self, exclude=[], scalling_method='standard') -> DataFrame:
        print("\n\nLe prétraitement a commencé!\n\n")
        
        print("Remplissage des valeurs nules..\n")
        self.__missing_values()
        print("Remplissage terminé!\n")
        
        print("Normalisation des données..\n")
        self.__scalling_data(method=scalling_method, exclude=exclude)
        
        print("Encodage des données..\n")
        self.__encoding_data(exclude)
        print("Encodage terminé!")
        
            
    def encoding_class(self, target_column):
        target_encoder = LabelEncoder()
        self.df[target_column] = target_encoder.fit_transform(self.df[target_column])

    def splitting_data(self, target_column):
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def head(self):
        print(self.df.head())
    
    def drop(self, columns=[]):
        self.df = self.df.drop(columns, axis=1)
    
    # Private methods
    def __missing_values(self):
        if self.type == 'csv':
            for column in self.df.columns:
                if is_numeric_dtype(self.df[column]):
                    self.df[column] = self.df[column].fillna(self.df[column].mean())
                else:
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        elif self.type == 'arff':
            for column in self.df.columns:
                if is_numeric_dtype(self.df[column]):
                    self.df[column] = self.df[column].replace({'?': np.nan}).astype(float)
                    self.df[column] = self.df[column].fillna(self.df[column].mean())
                else:
                    self.df[column] = self.df[column].replace({'?': np.nan})
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        else:
            raise Exception('Méthode non implémenté encore!')
        
        
    def __encoding_data(self, exclude):
        object_columns = self.df.select_dtypes(exclude=['number']).columns.tolist()
        
        if object_columns:
            ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
            if exclude:
                for col in exclude:
                    if col in object_columns:
                        object_columns.remove(col)
                        
            encoded_data = ohe.fit_transform(self.df[object_columns])
            self.df = concat([self.df, encoded_data], axis=1).drop(columns=object_columns)
        else:
            return

    def __scalling_data(self, exclude, method='standard'):
        numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_columns:
            if method == 'standard':
                scaler = StandardScaler()
                
                if exclude:
                    for col in exclude:
                        if col in numeric_columns:
                            numeric_columns.remove(col)
                        
                self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])

            else:
                raise Exception('Méthode non implémenté encore!')
        else:
            return
        