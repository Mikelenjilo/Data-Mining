from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter

def dbscan(df, eps, min_samples):
    max_silouhette = 0
    max_eps = 0
    max_min_sample = 0
    num_samples = df.shape[0]
    
    for ep in eps:
        for min_sample in min_samples:
            clusterer = DBSCAN(eps=ep, min_samples=min_sample)
            labels = clusterer.fit_predict(df)
            nb_clusters = len(Counter(labels).keys())
            
            if nb_clusters < 2 or nb_clusters > num_samples - 1:
                print("Erreur, essayer avec de nouvelles valeurs.")
                break
            
            silouhette = silhouette_score(df, labels)
            
            if silouhette > max_silouhette:
                max_silouhette = silouhette
                max_eps = ep
                max_min_sample = min_sample
                
    return {
        'algorithm': 'dbscan',
        'type': 'unsupervised', 
        'silouhette': float(max_silouhette),
        'eps': max_eps,
        'min_samples': max_min_sample,
    }