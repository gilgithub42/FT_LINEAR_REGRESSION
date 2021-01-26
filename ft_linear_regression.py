# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ft_linear_regression.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gigregoi <gigregoi@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/09/16 19:15:14 by gigregoi          #+#    #+#              #
#    Updated: 2020/12/24 06:04:06 by gigregoi         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from Utils.MyStatisticFunction import std_, mean_

class ft_linear_regression():
    '''
    Ce programme est écrit pour déterminer les paramètres de la régression multiple 
    d'une grandeur Y dont la valeur est fournie pour m exemples (échantillons)
     en fonction de (n-1) variables, fournies pour ces memes m exemples, ce qui 
     correspond à une matrice m * (n-1). En ajoutant la présence d'un biais,
     on obtient, X[m, n] et Y[n].
     Pour le cas d'application de la prédiction du prix d'une voiture en fonction
      de son kilométrage, n = 2 et la régression est qualifiée de "linéaire". 
     La régression s'écrit : prix = theta1 * km + theta0
     X : km
     y : prix

     L'objectif du programme est la détermination du vecteur theta (theta0, theta1, ...)

     Méthode imposée par le sujet 
     les moindres carrés avec résolution par la méthode du gradient

    Les bibliothèques numpy, pandas et matplotlib.pyplot sont utilisées pour :
    numpy : les calculs matriciels
    pandas : l'extraction des données au format tableau
    matplotlib : pour le rendu graphique
    Les outils de résolution ne sont pas utilisés.

    Le programme est organisé comme suit :
    - fonction def __init__() : générateur de la classe
    ''' 

    def __init__(self, theta):
        if isinstance(theta, (list, np.ndarray)):
            self.theta = theta
        else:
            print ("erreur sur les paramètres de la régression linéaire")
            return
        self.COSTvsITER = [] #sauvegarde de la convergence du cout vs iteration
    
    def predict_(self, x0, theta):

        if isinstance(x0, float):
            x0_tab = np.append([[1.]], [[x0]], axis = 1)
        if isinstance(x0, list):
            x0.insert(0,1)
            x0_tab = np.asarray(x0)
        if isinstance(x0, np.ndarray):
            x0_tab = np.asarray(x0)
            n = x0_tab.shape[0]
            OFFSET = np.ones((n,1))
            x0_tab = np.concatenate((OFFSET, x0), axis = 1)
        y0 = np.dot(x0_tab, theta.T)
        return y0
    
    def norm_(self, Vec):
        Vec_norm = (Vec - mean_(Vec))/std_(Vec)
        return Vec_norm
    
    def UnNorm_theta(self, theta, X, y):
        n = X.shape[1]
        UnTheta = np.zeros((1,n+1))
        sum = 0
        for i in range(0, n):
            sum += theta[0][i+1]*mean_(X[:,i])/std_(X[:,i])
            UnTheta[0][i+1] = theta[0][i+1]*std_(y)/std_(X[:,i])
        UnTheta[0][0] = (theta[0][0] - sum)*std_(y)+mean_(y)
        return (UnTheta)

    def __cost_function(self, X, Y, theta):
        m = X.shape[0]
        OFFSET = np.ones((m,1))
        X_off = np.concatenate((OFFSET, X), axis = 1) #Ajout de la colonne biais
        y_pred = np.dot(X_off, theta.T)
        cost = (1/(2 * m)) * np.dot((y_pred - Y).T,(y_pred - Y))
        return np.ndarray.item(cost)
    
    def grad_(self, X, y, theta, n):
        grad = 1/n*np.dot(X.T, (np.dot(X, self.theta.T)-y))
        return grad.T
    
    def fit(self, X, y,m,n,alpha = 0.001, n_cycle = 2000):
        float_formatter = "{:.2e}".format
        np.set_printoptions(formatter={'float_kind':float_formatter})
        y_norm = self.norm_(y)
        X_norm = np.ones((m,1))
        for i in range(0,n):
            temp = self.norm_(X[:,i]).reshape((m,1))
            X_norm = np.concatenate((X_norm, temp), axis = 1)
        for i in range(n_cycle):
            self.theta = self.theta - alpha/m*self.grad_(X_norm, y_norm, self.theta,n)
            # L'algorithme prevoit que le résultat des équations de la phase d'apprentissage (ci-dessus)
            # doit être stocké dans des variables temporaires temp_theta_j (j indice correspond 
            # à la variable associée) et qu'à chaque fin d'itération, temp_theta_j sont affectes
            # a theta_j pour la boucle suivante. Le fait d'utiliser une écriture matricielle rend cette étape inutile 

            if (i % int(round(n_cycle/1000)) == 0):
                UnTheta = self.UnNorm_theta(self.theta, X, y)
                self.COSTvsITER.append((i, self.__cost_function(X, y, UnTheta))) 
            if (i % int(round(n_cycle/10)) == 0):
                print(f"apres {i} iterations theta  = {UnTheta.flatten()} \n")
        ListITvsCost = list(zip(*self.COSTvsITER))
        plt.plot(ListITvsCost[0] , ListITvsCost[1] , "-+", label = "Cost Function")
        plt.legend()
        plt.xlabel('Iteration', fontsize = 10)
        plt.ylabel('Cost Function', fontsize = 10)
        plt.title('Convergence', fontsize = 16, horizontalalignment = 'left', loc = 'left')
        plt.grid(True)
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        return self.theta
    

    def Coefficient_determination(self, X, Y, theta,n):
        '''
        Description:
            Display the quality of the model.
        Args:
            X: has to be a numpy.ndarray, a matrix of dimension (number of training examples, number of features).
            Y: has to be a numpy.ndarray, a vector of dimension (number of training examples, 1).
        Returns:
            the determination coefficient
        Theory
            Variance equation
            yi - y_mean = (y_pred_i - y_mean) + (y_i - y_pred_i) 
            (yi - y_mean)² = (y_pred_i - y_mean)² + (y_i - y_pred_i)²
            ∑i=1->n[(yi - y_mean)²] = ∑i=1->n[(y_pred_i - y_mean)²] + ∑i=1->n[(y_i - y_pred_i)²]
            SCT : somme des carrés     SCE : Somme des carrés       SCR : Somme des carrés
                        totale                   expliquee                  résiduelle    
                    
            The Quality Determinant Coef R^2 = (SCE / SCT)
        Raises:
            This function should not raise any Exception.
        '''
        text =""
        if(n<10):
            for i in range(1,n):
                text += f"+ {theta.item(0,i):.10f} * X" + f"{i}"
            text = f" {theta.item(0,0):.10f}" + text
        E = Y - self.predict_(X, theta) # y_i - y_pred_i
        Y_moy = 1/len(Y) * sum(Y)
        SCT = sum((Y - Y_moy)**2)
        SCE = sum((self.predict_(X, theta) - Y_moy)**2)
        SCR = sum(E**2)
        output = f"\nL'equation de regression lineaire est  :" + text + "\n\n\n" + '\33[0m'
        output += f"Metrique de precision de la regression lineaire : Coefficient de correlation\n\n"
        output += f"Il est defini a partir de l'equation de la variance SCT = SCE + SCR\n"
        output += f"Somme des carrés totale     SCT = ∑i=1->n[(yi - y_mean)²]        : {SCT[0]:11.2f}\n"
        output += f"Somme des carrés expliquée  SCE = ∑i=1->n[(y_pred_i - y_mean)²]  : {SCE[0]:11.2f}\n"
        output += f"Somme des carrés résiduelle SCR = ∑i=1->n[(y_i - y_pred_i)²]     : {SCR[0]:11.2f}\n\n"
        output += f"==> Convergence si SCT = SCE + SCR         : {SCT[0]:11.2f} = {(SCR + SCE)[0]:.2f}\n\n"
        output += f"Le coefficient de correlation  R^2 = (SCE / SCT)   : {(SCE / SCT)[0]:12.2%}\n"
        R2 = f"R² = {(SCE / SCT)[0]:12.2%}"
        return output, text, R2
        
if __name__ == '__main__':

    chemin_csv = r"./Resources/data.csv"
    alpha = 0.001
    n_cycle = 20001

    try :
        choix = int(input("Choisissez 1 pour la partie obligatoire, 2 pour la partie BONUS - Default value = 1\n"))
        if (choix ==2):
            chemin_csv = r"./Resources/spacecraft_data.csv"
            alpha = 0.001
            n_cycle = 50001
    except :
        choix = 1

    try :
        data = pd.read_csv(chemin_csv, delimiter=',')
        var_name = []
        for col_name in data.columns: 
            var_name.append(col_name)
        m,n = data.shape
        print("Number of variables :", '\33[92m', f"{n-1:-10}", '\33[0m', "and Sample number : ", '\33[92m',f"{m:-10}\n", '\33[0m')
        for i,var in enumerate(var_name):
            print("variable X", f"{i} : ", f"{var}")
        print("-----------------------------------------------------------------")
        print("First data lines")
        print(data.head(5))
        plt.pause(2)
        if (n > 2):
            print("\n")
            print("Régression linéaire multiple")
    except ValueError:
        print("usage : Le fichier de donnees doit etre au format csv et contenir m lignes et n colonnes\n \
            la derniere colonne contient la variable a predire")
    data = np.array(data)
    X = np.array(data[:,:n-1])
    y = np.array(data[:,n-1])
    y = np.array(y).reshape(-1,1)

    if (n ==2):
        X = np.array(X).reshape(-1,1)
        y = np.array(y).reshape(-1,1)
    print("\n")
    print("-----------------------------------------------------------------")
    print("---------------------RESOLUTION----------------------------------")
    print("-----------------------------------------------------------------")
    theta = np.zeros((1,n))
    model = ft_linear_regression(theta)
    model.fit(X, y,m,n-1,alpha,n_cycle)
    UnTheta = model.UnNorm_theta(model.theta, X, y) # Theta denormalisé
    if (n==2):
        np.save('./Res1Var/theta_output', UnTheta)
        print('\33[92m \n\n\n*************************** Coefficients Storage   **************************************')
        print("******************  Regression Coefficients are saved to './Res1Var/theta_output'"'********')
        print('*****************************************************************************************')
        print('*****************************************************************************************\n\n\n\33[0m')
        time.sleep(2)
    if (n>2):
        np.save('./ResMultiVar/theta_output', UnTheta)
        print('\33[92m \n\n\n*************************** Coefficients Storage   ************************************')
        print("*****************  Regression Coefficients are saved to './ResMultiVar/theta_output'"'*****')
        print('*****************************************************************************************')
        print('*****************************************************************************************\n\n\n\33[0m')
        time.sleep(2)
    print("----------------------------------------------")
    print (model.Coefficient_determination(X,y,UnTheta,n)[0])

    OFFSET = np.ones((m,1))
    X_off = np.concatenate((OFFSET, X), axis = 1) #Ajout de la colonne biais
    y_pred = np.dot(X_off, UnTheta.T)

    if (n ==2):
        fig, axs = plt.subplots(figsize=(10,5))
        plt.title('Vehicle price vs mileage', fontsize = 16, horizontalalignment = 'left', loc = 'left')
        axs.set_xlabel(var_name[0], fontsize = 10)
        axs.set_ylabel(var_name[1], fontsize = 10)
        axs.plot(X, y, "bo", label = "Sell price")
        axs.plot(X, y_pred, "-.", label = "Predicted sell price")
        plt.grid(True)
        Eq_et_R2 = model.Coefficient_determination(X,y,UnTheta,n)[1] + "\n" + \
        model.Coefficient_determination(X,y,UnTheta,n)[2]
        axs.text(150000, 8000, Eq_et_R2, style='normal',
        bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 10})

        plt.show()
    
    if (n>2):
        qt = int(np.ceil(n**0.5))
        fig, axs = plt.subplots(qt,qt,figsize=(10,5))
        left   =  0.125  # the left side of the subplots of the figure
        right  =  0.9    # the right side of the subplots of the figure
        bottom =  0.1    # the bottom of the subplots of the figure
        top    =  0.9    # the top of the subplots of the figure
        wspace =  .5     # the amount of width reserved for blank space between subplots
        hspace =  0.8    # the amount of height reserved for white space between subplots

        # This function actually adjusts the sub plots using the above paramters
        plt.subplots_adjust(
            left    =  left, 
            bottom  =  bottom, 
            right   =  right, 
            top     =  top, 
            wspace  =  wspace, 
            hspace  =  hspace
        )
        fig.suptitle('Price vs each variable')
        for i in range(1,n):
            feature_name = var_name[i-1] 
            row = (i-1)%qt
            pos = (i-1)//qt
            axs[row][pos].set_xlabel(feature_name)
            axs[row][pos].set_ylabel(var_name[n-1])
            axs[row][pos].plot(X[:,i-1],y, linestyle='None', marker = 'o', color = 'orange')
            axs[row][pos].plot(X[:,i-1],y_pred, color = 'blue', linestyle='None', marker = 'o')
        line_labels = ['raw data', 'prediction']

        fig.legend(
            labels=line_labels,   # The labels for each line
            loc="center right",   # Position of legend
            borderaxespad=0.1,    # Small spacing around legend box
        )
        plt.show()
        fig1 = plt.figure(figsize=(10,5))
        left   =  0.125  # the left side of the subplots of the figure
        right  =  0.9    # the right side of the subplots of the figure
        bottom =  0.1    # the bottom of the subplots of the figure
        top    =  0.9    # the top of the subplots of the figure
        wspace =  .5     # the amount of width reserved for blank space between subplots
        hspace =  0.8    # the amount of height reserved for white space between subplots

        # This function actually adjusts the sub plots using the above paramters
        plt.subplots_adjust(
            left    =  left, 
            bottom  =  bottom, 
            right   =  right, 
            top     =  top, 
            wspace  =  wspace, 
            hspace  =  hspace
        )
        fig1.suptitle('Price vs variable - 3D')
        k = 1
        for i in range(1,n):
            
            for j in range(i+1,n):
                axs = fig1.add_subplot(qt, qt, k, projection='3d')
                k += 1
                feature_name1 = var_name[i-1]
                feature_name2 = var_name[j-1]
                row = (i-1)%qt+1
                pos = (i-1)//qt+1
                X1 = X[:,i-1]
                X2 = X[:,j-1]  
                axs.set_xlabel(feature_name1)
                axs.set_ylabel(feature_name2)
                axs.set_zlabel(var_name[n-1])
                axs.scatter(X1, X2, y, s = 2.5, marker='o', color = 'blue')
                axs.scatter(X1, X2, y_pred, s = 2.5, marker='+', color = 'orange')
        line_labels = ['raw data', 'prediction']
        fig1.legend(
            markerscale = 2,
            labels=line_labels,   # The labels for each line
            loc="center right",   # Position of legend
            borderaxespad=0.1,    # Small spacing around legend box
        )
        plt.show()