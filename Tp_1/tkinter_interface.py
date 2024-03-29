from tkinter import *
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd

def tp1(filepath):
    data_frame = pd.read_csv(filepath)
    # les 5 premières lignes du fichier
    title_label = Label(root, text='Les 5 premières lignes du fichier')
    title_label.pack()
    text_output = Text(root, height=10, width=100)
    text_output.pack(pady=10)
    text_output.delete("1.0", END)
    text_output.insert(END, data_frame.head())

    # nombre d'instances et de colonnes
    title_label = Label(root, text='Nombre d\'instances et de colonnes')
    title_label.pack()
    text_output = Text(root, height=1)
    text_output.pack(pady=10)
    text_output.delete("1.0", END)
    text_output.insert(END, data_frame.shape)

    # nom des attributs et leur nombre et leur type
    title_label = Label(root, text='Nom des attributs et leur nombre et leur type')
    title_label.pack()
    text_output = Text(root, height=10, width=100)
    text_output.pack(pady=10)
    text_output.delete("1.0", END)
    text_output.insert(END, 'Nom des attributs\n') 
    text_output.insert(END, '-----------------\n') 
    for i in data_frame.columns.tolist():
        text_output.insert(END, i + '\n')
    text_output.insert(END, '\n')
    text_output.insert(END, 'Nombre d\'attributs :' + str(len(data_frame.columns.tolist()))) 
    text_output.insert(END, '\n')
    text_output.insert(END, 'Type des attributs\n')
    text_output.insert(END, '------------------\n')
    for i in data_frame.dtypes.tolist():
        text_output.insert(END, str(i) + '\n')

    # les 5 nombres pour chaque attribut : le min, le max, le median, q1 et q3
    title_label = Label(root, text='Les 5 nombres pour chaque attribut : le min, le max, le median, q1 et q3')
    title_label.pack()
    text_output = Text(root, height=10, width=100)
    text_output.pack(pady=10)
    text_output.delete("1.0", END)
    text_output.insert(END, 'Les 5 nombres pour chaque attribut\n')
    text_output.insert(END, '---------------------------------\n')
    for i in data_frame.columns.tolist():
        a = data_frame[i]
        if a.dtype != 'object':
            text_output.insert(END, i + '\n')
            text_output.insert(END, 'Min: ' + str(data_frame[i].min()) + '\n')
            text_output.insert(END, 'Max: ' + str(data_frame[i].max()) + '\n')
            text_output.insert(END, 'Median: ' + str(data_frame[i].median()) + '\n')
            text_output.insert(END, 'Q1: ' + str(data_frame[i].quantile(0.25)) + '\n')
            text_output.insert(END, 'Q3: ' + str(data_frame[i].quantile(0.75)) + '\n')
            text_output.insert(END, '\n')
        else:
            text_output.insert(END, 'Impossible de caluler les 5 nombres a cause du type.')
            text_output.insert(END, '----------------------------------------------------')
    
    # boxplots de chaque attributs sur le meme graphe
    data_frame.boxplot()
    plt.xticks(rotation=45)
    plt.title('Boxplots de tous les attributs')
    plt.xlabel('Attributs')
    plt.ylabel('Valeurs')
    plt.show()    

    # scatter plot
    attributs = data_frame.columns

    for x in attributs:
        for y in attributs:
            if (x != y):
                x_column = data_frame[x]
                y_column = data_frame[y]
                plt.scatter(x_column, y_column, color='red')
                plt.xlabel(x_column.name)
                plt.ylabel(y_column.name)

                plt.tight_layout()
                plt.show()

    # mode, mean, median pour chaque attribut
    title_label = Label(root, text='Mode, mean, median pour chaque attribut')
    title_label.pack()
    text_output = Text(root, height=10, width=100)
    text_output.pack(pady=10)
    text_output.delete("1.0", END)
    for attribut in attributs:
        a = data_frame[attribut]
        if a.dtype != "object":
            text_output.insert(END, "\nAttribut:" + str(attribut) + '\n')
            text_output.insert(END, "\t- Mode:" + str(a.mode().values[0]) + '\n')
            text_output.insert(END, "\t- Mean:" + str(a.mean()) + '\n')
            text_output.insert(END, "\t- Median:" + str(a.median()) + '\n')
            text_output.insert(END, '----------------------------------------------------' + '\n')
        else:
            text_output.insert(END, 'Impossible de calculer le mode, le mean et le median de cette attribut')
            text_output.insert(END, '----------------------------------------------------')
    
    # valeurs manquantes
    for attribut in data_frame.columns:
        a = data_frame[attribut]
        if a.dtype == "object":
            data_frame.replace('?', a.mode().values[0], inplace=True)
        else:
            data_frame.replace('?', a.median(), inplace=True)

    # normaliser les donnees par MIN/MAX
    for attribut in attributs:
        a = data_frame[attribut]
        if data_frame[attribut].dtype != "object":
            min = a.min()
            max = a.max()
            a = (a - min) / (max - min)

    # normaliser les donnees par Z-score
    for attribut in attributs:
        a = data_frame[attribut]

        if a.dtype != "object":
            a = (a - a.mean()) / a.std()

    

def open_file():
    filepath = filedialog.askopenfilename(title='Choisir un fichier benchmark', filetypes=[("CSV Files", "*.csv")])
    if filepath:
        print('Fichier selectionnee: ', filepath)
        tp1(filepath)

root = Tk()
root.title("TP 1")

open_button = Button(root, text='Ouvrir', command=open_file)
open_button.pack(pady=10)

open_button = Button(root, text='Clustering', command=open_file)
open_button.pack(pady=10)

open_button = Button(root, text='Classification supervisee', command=open_file)
open_button.pack(pady=10)


root.mainloop()