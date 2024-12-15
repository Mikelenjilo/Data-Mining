from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans(df, nb_clusters=None, auto=False):
    if not auto:
        clusterer = KMeans(n_clusters=nb_clusters)
        labels = clusterer.fit_predict(df)

        silouhette = silhouette_score(df, labels)

        return {
            'algorithm': 'kmeans',
            'type': 'unsupervised', 
            'silouhette': silouhette,
            'k': nb_clusters,
        }
    else:
        max_silouhette = 0
        max_k = 0
        for i in range(2, 30):
            clusterer = KMeans(n_clusters=i)
            labels = clusterer.fit_predict(df)

            silouhette = silhouette_score(df, labels)
            
            if silouhette > max_silouhette:
                max_silouhette = silouhette
                max_k = i
                
                    
        return {
            'algorithm': 'kmeans',
            'type': 'unsupervised', 
            'silouhette': float(max_silouhette),
            'k': max_k,
        }
            
                
        
