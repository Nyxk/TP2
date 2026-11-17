import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# lecture de la base de données
df = pd.read_csv("sp.csv")
df = df.dropna() # pour supprimer les lignes avec des valeurs manquantes
print(df.head()) # pour afficher les premières lignes
df = df.sample(frac=1)

# matrice des entrées (longueur du bec, largueur du bec, longueur des nageoires, poids du corps)
entrees = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]].to_numpy()

# matrice des sorties (manchot Adélie, manchot Papou, manchot à jugulaire)
sp = ["Adelie","Gentoo","Chinstrap"]
sorties = np.zeros([len(df),len(sp)])
for s in range(len(sp)):
   sorties[:,s] = df.species==sp[s]



df_test = pd.read_csv("testset.csv")
df_test = df_test.dropna()
df_test = df_test.sample(frac=1)
entrees_test = df_test[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]].to_numpy()
sp_test = ["Adelie","Gentoo","Chinstrap"]
sorties_test = np.zeros([len(df_test),len(sp)])

for s in range(len(sp_test)):
   sorties_test[:,s] = df_test.species==sp[s]


def tester(entrees_test, sorties_test, matrice, entree_min, entree_max):
    x = len(sorties_test[0])
    y = len(entrees_test[0]) + 1
    err_test = 0
    entrees_test_norm =(entrees_test-entree_min) / (entree_max-entree_min)
    for example in range(len(entrees_test_norm)):
        pattern = list(entrees_test_norm[example]) +[1]
        S=[]
        for j in range(x):
            somm= sum(matrice[j][i] * pattern[i] for i in range(y))
            S.append(1 if somm >= 0 else 0)
        for i in range(x):
            err_test += abs(sorties_test[example][i] - S[i])
    return err_test
    

##normaliser les entrees
def hoff(entrees, sortie_attendu, entrees_test, sorties_test):
    x = len(sortie_attendu[0])
    eta = 0.01
    #4 attributs + 1 biais 
    y = len(entrees[0]) +1
    matrice = np.random.uniform(1, 10, size=(x, y))
    #Normalisation entre 0 et 1 
    entree_min, entree_max = entrees.min(axis = 0), entrees.max(axis = 0)
    entree_norm =(entrees - entrees.min(axis = 0)) /(entrees.max(axis = 0) - entrees.min(axis = 0))
    erreurs = []
    erreurs_test = []
    #Nombre d'etapes choisit par l'utilisateur
    for etape in range(500):
        err_etape = 0
        for example in range(len(entree_norm)):
            pattern = list(entree_norm[example]) + [1] 
            S=[]
            for j in range(x):
                somm = sum(matrice[j][i] * pattern[i] for i in range(y))
                S.append(1 if somm >= 0 else 0 )
            for i in range(y):
                for j in range(x):
                    err = sortie_attendu[example][j] - S[j]
                    matrice[j][i] += eta * err * pattern[i]
            for i in range(x):
                err_etape += abs(sortie_attendu[example][i] -S[i])
        erreurs.append(err_etape)
        erreurs_test.append(tester(entrees_test, sorties_test, matrice,entree_min,entree_max))
    return matrice, erreurs, erreurs_test






a = hoff(entrees,sorties, entrees_test, sorties_test)
x = [x for x in range(500)]
plt.plot(x,a[1], label = "Learning model")
plt.plot(x,a[2], label = "Test model :))")
plt.title("Pour η = 0.01")
plt.xlabel("Etapes")
plt.ylabel("Erreur")
plt.legend()
plt.show()



