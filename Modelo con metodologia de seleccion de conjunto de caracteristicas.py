import pandas as pd
import numpy as np
import statsmodels.api as sm
import sklearn.ensemble as ske
import joblib
import pickle

from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



def leerDatos():
    data = pd.read_csv('data.csv', sep='|')
    
    X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
    y = data['legitimate'].values
    return X, y, data

def crearArreglos(n_subconjunto):
    arreglo = []
    sublista = []
    for i in range(int(n_subconjunto)):
        sublista = []
        arreglo.append(sublista)
        
    return arreglo

def select_features_chi2(X_train, y_train, data, contador):
    """
    listaChi2 = ["SizeOfCode", "SizeOfInitializedData", "SizeOfUninitializedData", "BaseOfCode", "ImageBase", "CheckSum"
                  , "SizeOfStackReserve", "NumberOfRvaAndSizes", "SectionsMinRawsize"]
    """
    listaChi2 = []
    
    # configure to select all features
    fsChi2 = SelectKBest(score_func=chi2, k=14)
    # learn relationship from training data
    fsChi2.fit(X_train, y_train)
    # transform train input data
    X_train_chi2 = fsChi2.transform(X_train)
    # transform test input data
    #X_test_chi2 = fsChi2.transform(X_test)
    
    indices = np.argsort(fsChi2.scores_)[::-1][:fsChi2.k]
    
    for f in sorted(np.argsort(fsChi2.scores_)[::-1][:fsChi2.k]):
        listaChi2.append(data.columns[2+f])
        """
        print(data.columns[2+f])
        print(X_train_chi2[f])
        """
    
    contador = contador + 1
    return listaChi2, X_train_chi2, contador

def select_features_f_classif(X_train, y_train, data, contador):
    """
    listasF_classif=["Machine", "SizeOfOptionalHeader",  "MajorOperatingSystemVersion",
                      "MajorSubsystemVersion", "Subsystem", "DllCharacteristics", "SizeOfStackReserve",
                       "SectionsMeanEntropy", "SectionsMaxEntropy"]
    """
    listasF_classif=[]
    fs_f_classif = SelectKBest(score_func=f_classif, k=11)
    fs_f_classif.fit(X_train, y_train)
    X_train_f_classif = fs_f_classif.transform(X_train)
    #X_test_f_classif = fs_f_classif.transform(X_test)
    
    indices = np.argsort(fs_f_classif.scores_)[::-1][:fs_f_classif.k]
    for f in sorted(np.argsort(fs_f_classif.scores_)[::-1][:fs_f_classif.k]):
        listasF_classif.append(data.columns[2+f])
        
    
    contador = contador + 1
    return listasF_classif, X_train_f_classif, contador

def select_features_mutual_info_classif(X_train, y_train, data, contador):
    """
    listFs_mutual = ["Characteristics", "AddressOfEntryPoint", "SectionsMinEntropy", "SectionsMaxEntropy"
                     , "SectionsMeanVirtualsize", "SectionsMinVirtualsize", "SectionMaxVirtualsize"
                     ,"ResourcesMinEntropy", "ResourcesMaxEntropy", "ResourcesMeanSize", "ResourcesMaxSize"]
    """
    listFs_mutual = []
    
    # configure to select all features
    fs_mutual = SelectKBest(score_func=mutual_info_classif, k=11)
    # learn relationship from training data
    fs_mutual.fit(X_train, y_train)
    # transform train input data
    X_train_mutual = fs_mutual.transform(X_train)
    # transform test input data
    #X_test_mutual = fs_mutual.transform(X_test)
    
    indices = np.argsort(fs_mutual.scores_)[::-1][:fs_mutual.k]
    for f in sorted(np.argsort(fs_mutual.scores_)[::-1][:fs_mutual.k]):
        listFs_mutual.append(data.columns[2+f])
        

    contador = contador + 1
    return listFs_mutual, X_train_mutual, contador




def llenarListaPreSeleccion(listaClasificadores, lista, X_new_chi2, X_new_classif, X_new_mutual):
    index = 0
    bandera = False
    contador = False
    matrizX_new_SubConjunto0 = pd.DataFrame()
    matrizX_new_SubConjunto1 = pd.DataFrame()
    matrizX_new_SubConjunto2 = pd.DataFrame()
    matrizX_new_SubConjunto3 = pd.DataFrame()
    matrizX_new_SubConjunto4 = pd.DataFrame()
    
    for i in range(len(listaClasificadores)):
        n = len(listaClasificadores[i])
        
        
        if contador == False:
            contador = True
            for k in range(n):
                if index < n_subconjunto and bandera == False:
                    print("listaClasificadores["+str(i)+"]["+str(k)+"] a Lista["+str(index)+"]")
                    lista[index].append(listaClasificadores[i][k])
                    if i == 0:
                        #matrizX_new_SubConjunto[listaClasificadores[i][k]] = X_new_chi2[:,k]
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_chi2[:,k]'.format(index))
                    elif i == 1:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_classif[:,k]'.format(index))
                    elif i == 2:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_mutual[:,k]'.format(index))

                   
                    index = index + 1
                    
                elif index == n_subconjunto or bandera == True:
                    if bandera == False:
                        index = index - 1
                    print("listaClasificadores["+str(i)+"]["+str(k)+"] a Lista["+str(index)+"]")
                    lista[index].append(listaClasificadores[i][k])
                    if i == 0:
                        #matrizX_new_SubConjunto[listaClasificadores[i][k]] = X_new_chi2[:,k]
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_chi2[:,k]'.format(index))
                    elif i == 1:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_classif[:,k]'.format(index))
                    elif i == 2:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_mutual[:,k]'.format(index))
                   
                    index = index - 1
                    if index < 0:
                        bandera = False
                        index = index + 1
                    else:
                        bandera = True
                        
                elif index < 0 and bandera == False:
                    print("listaClasificadores["+str(i)+"]["+str(k)+"] a Lista["+str(index+1)+"]")
                    lista[index+1].append(listaClasificadores[i][k])
                    if i == 0:
                        #matrizX_new_SubConjunto[listaClasificadores[i][k]] = X_new_chi2[:,k]
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_chi2[:,k]'.format(index+1))
                    elif i == 1:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_classif[:,k]'.format(index+1))
                    elif i == 2:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_mutual[:,k]'.format(index+1))
                    
                    index = index + 1
    
        
        elif contador == True:
            contador = False
            for k in reversed(range(n)):
                if index < n_subconjunto and bandera == False:
                    print("listaClasificadores["+str(i)+"]["+str(k)+"] a Lista["+str(index)+"]")
                    lista[index].append(listaClasificadores[i][k])
                    if i == 0:
                        #matrizX_new_SubConjunto[listaClasificadores[i][k]] = X_new_chi2[:,k]
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_chi2[:,k]'.format(index))
                    elif i == 1:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_classif[:,k]'.format(index))
                    elif i == 2:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_mutual[:,k]'.format(index))
                    
                    index = index + 1
                    
                elif index == n_subconjunto or bandera == True:
                    if bandera == False:
                        index = index - 1
                    print("listaClasificadores["+str(i)+"]["+str(k)+"] a Lista["+str(index)+"]")
                    lista[index].append(listaClasificadores[i][k])
                    if i == 0:
                        #matrizX_new_SubConjunto[listaClasificadores[i][k]] = X_new_chi2[:,k]
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_chi2[:,k]'.format(index))
                    elif i == 1:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_classif[:,k]'.format(index))
                    elif i == 2:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_mutual[:,k]'.format(index))
                    
                    index = index - 1
                    if index < 0:
                        bandera = False
                        index = index + 1
                    else:
                        bandera = True
                        
                elif index < 0 and bandera == False:
                    print("listaClasificadores["+str(i)+"]["+str(k)+"] a Lista["+str(index+1)+"]")
                    lista[index+1].append(listaClasificadores[i][k])
                    if i == 0:
                        #matrizX_new_SubConjunto[listaClasificadores[i][k]] = X_new_chi2[:,k]
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_chi2[:,k]'.format(index+1))
                    elif i == 1:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_classif[:,k]'.format(index+1))
                    elif i == 2:
                        exec('matrizX_new_SubConjunto{}[listaClasificadores[i][k]] = X_new_mutual[:,k]'.format(index))
                    
                    index = index + 1

        index = 0
        bandera = False
        print("\n")
    return lista, matrizX_new_SubConjunto0, matrizX_new_SubConjunto1, matrizX_new_SubConjunto2, matrizX_new_SubConjunto3, matrizX_new_SubConjunto4

def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features


    
contador = 0
X, y, data = leerDatos()


n_caracteristicas = X.shape[1]
#n_subconjunto = round(n_caracteristicas / 11)
n_subconjunto = 3

lista = crearArreglos(n_subconjunto)

listaChi2, X_new_chi2, contador = select_features_chi2(X, y, data , contador)
listasF_classif, X_new_classif, contador = select_features_f_classif(X, y, data, contador)
listFs_mutual, X_new_mutual, contador = select_features_mutual_info_classif(X, y, data, contador)

listaClasificadores = [listaChi2, listasF_classif, listFs_mutual]

preSeleccion, matrizX_new_SubConjunto0, matrizX_new_SubConjunto1, matrizX_new_SubConjunto2, matrizX_new_SubConjunto3, matrizX_new_SubConjunto4 = llenarListaPreSeleccion(listaClasificadores, lista, X_new_chi2, X_new_classif, X_new_mutual )

listaFinal = crearArreglos(n_subconjunto)

for i in range(int(n_subconjunto)):
    arreglo = []
    #exec('print(matrizX_new_SubConjunto{})'.format(i))
    exec('arreglo = forward_selection(matrizX_new_SubConjunto{}, y)'.format(i))
    for j in range(len(arreglo)):
        listaFinal[i].append(arreglo[j])


listaCaracteristicas = []
for f in range(len(listaFinal)):
    for g in range(len(listaFinal[f])):
        listaCaracteristicas.append(listaFinal[f][g])


print("\n")
listaCaracteristicas = list(set(listaCaracteristicas))

print("El nuevo tamaño es: "+str(len(listaCaracteristicas)))
for t in range(len(listaCaracteristicas)):
    print(" "+str(t+1)+" "+listaCaracteristicas[t])
    
matriz = pd.DataFrame()
for i in range(len(listaCaracteristicas)):
    matriz[listaCaracteristicas[i]] = data[listaCaracteristicas[i]]

X_train, X_test, y_train, y_test = train_test_split(matriz, y ,test_size=0.25, random_state=1)    



#Comparacion de algortimos
algorithms = {
        "DecisionTree": tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=25,
                                                    min_samples_split=2, min_samples_leaf=1,
                                                    min_weight_fraction_leaf=0.0, max_features=12),
        "RandomForest": ske.RandomForestClassifier(n_estimators=40, criterion="gini", max_depth=25, min_samples_split=2,
                                                   min_samples_leaf=1, max_features=6),
        "GradientBoosting": ske.GradientBoostingClassifier(loss="deviance", learning_rate=1,
                                                           n_estimators=40, max_depth=4, min_samples_split=2,
                                                           min_samples_leaf=1, max_features=6),
        "AdaBoost": ske.AdaBoostClassifier(n_estimators=85, learning_rate=1)
    }

results = {}
print("\nAhora probamos los algoritmos")
 
# Entrenamiento de algoritmos de clasificación al conjunto de entrenamiento
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    res = clf.predict(X_test)
    

    mt = confusion_matrix(y_test, res)
    
    exactitud = accuracy_score(y_test, res)
    sensibilidad = recall_score(y_test, res)
    especificidad = mt[0][0]/(mt[0][0]+mt[0][1])

    puntaje = f1_score(y_test, res)
    
    print("%s : %f %%" % (algo, score*100))


    

    
    print(mt)

    print("Falsos positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
    print('Falsos negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))

    print('exactitud : '+str(exactitud*100))

 
    print('sensibilidad : '+str(sensibilidad*100))


    print('especificidad : '+str(especificidad*100))

    print('Puntaje : '+str(puntaje*100))
    
    print("\n")

    results[algo] = score

winner = max(results, key=results.get)
print('\nEl algoritmo ganador es %s con un %f %% éxito' % (winner, results[winner]*100))
"""
# Guardando el algoritmo y la lista de funciones para futuras predicciones
print('Guardando el algoritmo y la lista de características en el directorio del clasificador...')
joblib.dump(algorithms[winner], 'classifier/classifier.pkl')
open('classifier/features.pkl', 'wb').write(pickle.dumps(listaCaracteristicas))
print('Saved')
"""
# Predecir los resultados del conjunto de prueba
y_pred = clf.predict(X_test)

