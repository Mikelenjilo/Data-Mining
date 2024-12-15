import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
from tkinter.ttk import Combobox
from models.df import DF
from algorithms.unsupervised.kmeans import kmeans
from algorithms.unsupervised.kmedoid import kmedoid
from algorithms.unsupervised.agnes import agnes
from algorithms.unsupervised.dbscan import dbscan
from algorithms.supervised.knn import knn
from algorithms.supervised.naive_bayes import naive_bayes
from algorithms.supervised.decision_tree import decision_tree
from algorithms.supervised.svm import svm
from algorithms.supervised.dnn import dnn

df = DF()
target_column = None

def load_dataset():
    dataset_path = filedialog.askopenfilename(
        title="Selectionner un Dataset",
        filetypes=[("Fichiers CSV", "*.csv"), ("Fichiers ARFF", "*.arff")]
    )

    if not dataset_path:
        return

    if dataset_path.endswith('.csv'):
        df.reading_data(dataset_path, file_type='csv')
        messagebox.showinfo("Dataset Chargé", f"Le dataset '{os.path.basename(dataset_path)}' a été chargé avec succès !")
        display_data(df.df)
    elif dataset_path.endswith('.arff'):
        df.reading_data(dataset_path, file_type='arff')
        messagebox.showinfo("Dataset Chargé", f"Le dataset '{os.path.basename(dataset_path)}' a été chargé avec succès !")
        display_data(df.df)
    else:
        messagebox.showerror("Erreur", "Format du fichier non pris en charge")
        
    select_target_column()
        
        
def display_data(data):
    for widget in frame_data.winfo_children():
        widget.destroy()

    preview = data.head()
    preview_text = preview.to_string(index=False)
    label_data = tk.Label(frame_data, text=preview_text, font=("Arial", 10), justify="left")
    label_data.pack(pady=10)
    
    if clear_button.winfo_ismapped() == False:
        clear_button.pack(pady=10)
        
def clear_data():
    for widget in frame_data.winfo_children():
        widget.destroy()

    clear_button.pack_forget()
    
def unsupervised_classification():
    window = tk.Toplevel(root)
    window.title("Classification Non Supervisée")
    window.geometry("400x400")

    tk.Label(window, text="Choisissez l'algorithme de classification non supervisée", font=("Arial", 12)).pack(pady=10)

    def kmeans_fun():
        k = simpledialog.askinteger("K-Means", "Entrez la valeur de k:")
        res = kmeans(df.df, nb_clusters=k)
        display_algorithm_results(res, result_label, algo='kmeans')

    def kmedoid_fun():
        k = simpledialog.askinteger("K-Medoid", "Entrez la valeur de k:")
        res = kmedoid(df.df, nb_clusters=k)
        display_algorithm_results(res, result_label, algo='kmedoid')

    def agnes_fun():
        k = simpledialog.askinteger("AGNES", "Entrez le nombre de clusters:")
        res = agnes(df.df, nb_clusters=k)
        display_algorithm_results(res, result_label, algo='agnes')

    def dbscan_fun():
        eps = simpledialog.askfloat("DBSCAN", "Entrez la valeur de eps:")
        min_pts = simpledialog.askinteger("DBSCAN", "Entrez la valeur de min_pts:")
        res = dbscan(df.df, eps=[eps], min_samples=[min_pts])
        display_algorithm_results(res, result_label, algo='dbscan')

    def choose_best():
        res_kmeans = kmeans(df.df, auto=True)
        res_kmedoid = kmedoid(df.df, auto=True)
        res_agnes = agnes(df.df, auto=True)
        res_dbscan = dbscan(df.df, eps=[0.1, 0.2, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], min_samples=[2, 3, 5, 10, 20, 25, 30])
        
        results = [res_kmeans, res_kmedoid, res_agnes, res_dbscan]

        best_algorithm = max(results, key=lambda x: x['silouhette'])

        result_text = (
            f"Meilleur algorithme:\n"
            f"Algorithme = {best_algorithm['algorithm']}\n"
            f"Score Silhouette = {best_algorithm['silouhette']:.4f}"
        )
        
        messagebox.showinfo("Meilleur Algorithme", result_text)
        
        

    tk.Button(window, text="K-Means", command=kmeans_fun).pack(pady=5)
    tk.Button(window, text="K-Medoid", command=kmedoid_fun).pack(pady=5)
    tk.Button(window, text="AGNES", command=agnes_fun).pack(pady=5)
    tk.Button(window, text="DBSCAN", command=dbscan_fun).pack(pady=5)
    tk.Button(window, text="Choisir le meilleur", command=choose_best).pack(pady=5)
    
    result_label = tk.Label(window, text="", font=("Arial", 12), justify="left")
    result_label.pack(pady=20)

def supervised_classification():
    window = tk.Toplevel(root)
    window.title("Classification Supervisée")
    window.geometry("400x400")

    tk.Label(window, text="Choisissez l'algorithme de classification supervisée", font=("Arial", 12)).pack(pady=10)


    def knn_fun():
        k = simpledialog.askinteger("KNN", "Entrez la valeur de k:")
        df.splitting_data(target_column)
        res = knn(df, k)
        display_algorithm_results(res, result_label, algo='knn')

    def naive_bayes_fun():
        df.splitting_data(target_column)
        res = naive_bayes(df)
        display_algorithm_results(res, result_label, algo='naive')

    def decision_tree_fun():
        df.splitting_data(target_column)
        res = decision_tree(df)
        display_algorithm_results(res, result_label, algo='tree')

    def svm_fun():
        df.splitting_data(target_column)
        res = svm(df)
        display_algorithm_results(res, result_label, algo='svm')

    def dnn_fun():
        num_layers = simpledialog.askinteger("Deep Neural Network", "Entrez le nombre de couches:")
        num_nodes = simpledialog.askinteger("Deep Neural Network", "Entrez le nombre de noeuds par couche:")
        df.encoding_class(target_column)
        df.splitting_data(target_column)
        res = dnn(df, [num_layers], [num_nodes], target_column)
        display_algorithm_results(res, result_label, algo='dnn')

    def choose_best():
        df.splitting_data(target_column)
        res_knn = knn(df, auto=True)
        res_naive = naive_bayes(df)
        res_tree = decision_tree(df)
        res_svm = svm(df)
        df.encoding_class(target_column)
        df.splitting_data(target_column)
        res_dnn = dnn(df, nb_hidden_layers=[2, 4, 6], nb_nodes=[4, 8, 16], target_column=target_column)
        
        results = [res_knn, res_naive, res_tree, res_svm, res_dnn]

        best_algorithm = max(results, key=lambda x: x['accuracy'])

        result_text = (
            f"Meilleur algorithme:\n"
            f"Algorithme = {best_algorithm['algorithm']}\n"
            f"Accuracy = {best_algorithm['accuracy']:.4f}"
        )
        
        messagebox.showinfo("Meilleur Algorithme", result_text)

    tk.Button(window, text="KNN", command=knn_fun).pack(pady=5)
    tk.Button(window, text="Naive Bayes", command=naive_bayes_fun).pack(pady=5)
    tk.Button(window, text="Decision Tree", command=decision_tree_fun).pack(pady=5)
    tk.Button(window, text="SVM", command=svm_fun).pack(pady=5)
    tk.Button(window, text="Deep Neural Network", command=dnn_fun).pack(pady=5)
    tk.Button(window, text="Choisir le meilleur", command=choose_best).pack(pady=5)
    
    result_label = tk.Label(window, text="", font=("Arial", 12), justify="left")
    result_label.pack(pady=20)


def display_algorithm_results(results, result_label, algo):
    if algo == 'kmeans' or algo == 'kmedoid' or algo == 'agnes':
        result_text = (
            f"Résultat:\n"
            f"Algorithme = {results['algorithm']}\n"
            f"Nombre de clusters = {results['k']}\n"
            f"Score Silhouette = {results['silouhette']:.4f}"
        )
    elif algo == 'dbscan':
        result_text = (
            f"Résultat:\n"
            f"Algorithme = {results['algorithm']}\n"
            f"Epsilon = {results['eps']}\n"
            f"Min points = {results['min_samples']:.4f}\n"
            f"Score Silhouette = {results['silouhette']:.4f}"
        )
    elif algo == 'knn':
        result_text = (
            f"Résultat:\n"
            f"Algorithme = {results['algorithm']}\n"
            f"K = {results['k']}\n"
            f"Accuracy = {results['accuracy'] * 100}%"
        )
    elif algo == 'naive' or algo == 'tree' or algo == 'svm':
        result_text = (
            f"Résultat:\n"
            f"Algorithme = {results['algorithm']}\n"
            f"Accuracy = {results['accuracy'] * 100}%"
        )
    elif algo == 'dnn':
        result_text = (
            f"Résultat:\n"
            f"Algorithme = {results['algorithm']}\n"
            f"Accuracy = {results['accuracy'] * 100}%\n"
            f"Nb couche cachées = {results['nb hidden layers']}\n"
            f"Nb noeuds = {results['nb nodes per hidden layer']}"
        )
        
    result_label.config(text=result_text)
    
def select_target_column():
    if df.df is None or df.df.empty:
        messagebox.showerror("Erreur", "Aucun dataset disponible.")
        return

    window = tk.Toplevel(root)
    window.title("Sélection de l'attribut cible")
    window.geometry("400x200")

    tk.Label(window, text="Sélectionnez l'attribut cible pour la classification supervisée :", font=("Arial", 12)).pack(pady=10)

    columns = df.df.columns.tolist()
    
    target_combobox = Combobox(window, values=columns, state="readonly", width=30)
    target_combobox.pack(pady=10)

    def set_target():
        global target_column
        target_column = target_combobox.get()
        if target_column:
            messagebox.showinfo("Attribut cible", f"L'attribut cible sélectionné est : {target_column}")
            window.destroy()
        else:
            messagebox.showerror("Erreur", "Veuillez sélectionner une colonne.")

    tk.Button(window, text="Confirmer", command=set_target).pack(pady=10)

def preprocess_data():
    if df.df is None or df.df.empty:
        messagebox.showerror("Erreur", "Aucun dataset chargé. Veuillez charger un dataset avant le prétraitement.")
        return

    if target_column is None:
        messagebox.showerror("Erreur", "Veuillez sélectionner une colonne cible avant le prétraitement.")
        return

    df.preprocessing(exclude=[target_column])

    messagebox.showinfo("Prétraitement", "Le prétraitement s'est terminé avec succès.")



root = tk.Tk()
root.title("Projet data mining")
root.geometry("1000x800")

label = tk.Label(root, text="Bienvenu au projet Data Mining", font=('Arial', 20))
label.pack(pady=20)

button = tk.Button(root, text="Charger un dataset", font=("Arial", 12), command=load_dataset)
button.pack(pady=10)

preprocess_button = tk.Button(root, text="Prétraitement", font=("Arial", 12), command=preprocess_data)
preprocess_button.pack(pady=10)

button_unsupervised = tk.Button(root, text="Classification Non Supervisée", font=("Arial", 12), command=unsupervised_classification)
button_unsupervised.pack(pady=10)

button_supervised = tk.Button(root, text="Classification Supervisée", font=("Arial", 12), command=supervised_classification)
button_supervised.pack(pady=10)

label_result = tk.Label(root, text="", font=("Arial", 12), justify="left")
label_result.pack(pady=20)

frame_data = tk.Frame(root)
frame_data.pack(pady=5)
clear_button = tk.Button(root, text="Effacer", font=("Arial", 12), command=clear_data)

root.mainloop()
