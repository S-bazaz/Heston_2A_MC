# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 13:16:12 2022

@author: samuel bazaz
"""

# !!!!!!!!!!!!!!!!!!!!!!!!!A MODIFIER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mainpath = "../"
# !!!!!!!!!!!!!!!!!!!!!!!!!A MODIFIER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# _____________________________packages______________________________________________


import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import time as ti
from tqdm import tqdm 
from scipy.stats import qmc

# parallélisation 
from numba import jit 

# _____________________________variables fixent de l'étude______________________________

# équations et définition du prix de l'option 
r = 0.03
rho = 0.2
theta = 0.3
xi = 0.5
K = 2
r = 0.03
rho = 0.2
k = 2

# choix d'étude
Y0 = np.array([20,50]) # conditions initiales [ Vt, St]
T = 5                  # temps d'étude des variables 

#_____________________________Fonctions de générations de Browniens______________________________

@jit
def W_simple(  T, n ):
  # generation simple par incrément gaussien
  L_T = np.linspace(0,T,n)
  W  =  np.zeros(n)
  for i in range(1,n):
    W[i] =  np.random.normal(W[i-1], T/n) 
  return  L_T, W


@jit
def W_bridge_dya(  T, n ): 
  # méthode Bridge en place: impose une taille de simulation en 2**m+1
  # une autre méthode avait été testé sans contrainte de format 
  # Mais la recherche des points précédents et suivants étaient trop couteux.
  
  m = len((bin(n)[2:])) -1        # on calcul l'exposant de 2 inférieur le plus proche 
  L_T = np.linspace(0, T, 2**m+1) # on initialise les objets au bon format
  W  =  np.zeros(2**m+1) 
  W[-1] =  np.random.normal(0, np.sqrt(T)) 
  for k in range(m):
    for l in range(2**k):
      # Suite a un calcul à la main et une preuve par récurrence, on a une formule explicite des 
      # indices à utiliser pour une itération 
      i     = (2*l+1)*(2**(m-k-1))  # indice de l'élément à modifier 
      i_inf = (2*l  )*(2**(m-k-1))  # indice de l'élément déjà calculé précédent
      i_sup = (2*l+2)*(2**(m-k-1))  # indice de l'élément déjà calculé suivant
      t_inf, t_sup, t = L_T[i_inf], L_T[i_sup], L_T[i]
      mu = ((t_sup-t)*W[i_inf]  + (t-t_inf)*W[i_sup] )/( t_sup - t_inf)
      sigma = np.sqrt( (t_sup-t)*(t-t_inf)/(t_sup - t_inf) ) 
      W[i] = np.random.normal( mu, sigma )
  return  L_T, W

@jit
def W_mlmc(T, etape , fonction_W, M=4): 

  # etape : prend les valeur de 1, 2,... désigne de manière indirecte le pas temporel fin (h_l)
  # Cette variable ne prend pas la valeur 0 ! 
  # On renvoie une liste de listes[indice de temps, valeur du brownien associée] caclulés avec le pas fin et le pas grossier
  # (1/n) désigne ici les pas temporels (de plus en plus fin)
  # M désigne par combien on subdivise les segments à l'étape suivante (opti selon l'article : M=4)
  # fonction_W : W_simple ou W_dya, si W_dya -> fixer M = 4 sinon M libre

  # Boucle W_dya
  if (fonction_W == W_bridge_dya):

    # Génération du Brownien de 1ère étape : pas grossier
    n1 = int((T-1)*M**(etape-1) + 1)
    liste_coord_Browniens = W_bridge_dya(T, n1)

    # Génération de l'échelle de temps plus fine
    L_T_fin = np.linspace(0, T, (T-1)*M**etape + 1)

    # On va générer un brownien plus fin à partir de celui-ci
    # Donc pour chacun des Browniens, on fait ce qui suit
    nb_iter = len(liste_coord_Browniens[0])-1
    W  =  np.zeros((T-1)*M**etape + 1)
    W[0] = liste_coord_Browniens[1][0]

    # Pour chaque sous-intervalle du Brownien considéré, on recalcule un Brownien
    # On parcourt chaque point du Brownien précédent et on génère M nouveaux points entre
    for j in range (nb_iter):
      W[4*(j+1)] = liste_coord_Browniens[1][j+1]
      n = M
      m = len((bin(n)[2:])) -1

      for k in range(m):
        for l in range(2**k):
          i     = (2*l+1)*(2**(m-k-1))
          i_inf = (2*l  )*(2**(m-k-1))
          i_sup = (2*l+2)*(2**(m-k-1))
          t_inf, t_sup, t = L_T_fin[4*j+i_inf], L_T_fin[4*j+i_sup], L_T_fin[4*j+i]
          mu = ((t_sup-t)*W[4*j+i_inf]  + (t-t_inf)*W[4*j+i_sup] )/( t_sup - t_inf)
          sigma = np.sqrt( (t_sup-t)*(t-t_inf)/(t_sup - t_inf) ) 
          W[4*j+i] = np.random.normal( mu, sigma )

    return  liste_coord_Browniens[0], liste_coord_Browniens[1], L_T_fin, W
  
  # Boucle W_simple
  else:

    # Génération du Brownien de 1ère étape : pas FIN
    n1 = int(2**etape)
    liste_coord_Browniens = fonction_W(T, n1)

    # Génération du Brownien de 2ème étape : pas GROSSIER
    n2 = n1//2
    L_T_grossier = np.zeros(n2)
    W_grossier = np.zeros(n2)

    for i in range(n2):
      L_T_grossier[i] = (liste_coord_Browniens[0][2*i] + liste_coord_Browniens[0][2*i+1])/2
      W_grossier[i] = (liste_coord_Browniens[1][2*i] + liste_coord_Browniens[1][2*i+1])/2

    return  L_T_grossier, W_grossier, liste_coord_Browniens[0], liste_coord_Browniens[1]

@jit
def stratification(n):
    # calcule une suite d'uniforme répartis sur des sous intervales de [0,1]
    U = np.zeros(n)
    for i in range(n):
        U[i]= np.random.uniform(i/n, (i+1)/n)
    np.random.shuffle(U)
    return U


@jit
def Bmuller(U1, U2):
    # méthode de conversion des uniformes en normales
    Z1 = np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)
    Z2 = np.sqrt(-2*np.log(U2))*np.sin(2*np.pi*U1)
    return Z1, Z2

# Afin d'utiliser le même pipline pour les méthodes QMC
# on définit des fonctions fantômes pour la création des noms de fonctions 
@jit
def W_simple_QMC(T, n):
    L_T = np.linspace(0,T,n)
    W   =  np.zeros(n)
    return  L_T, W

@jit
def W_simple_QMCHM(T, n):
    L_T = np.linspace(0,T,n)
    W   =  np.zeros(n)
    return  L_T, W

# ______________________________création des listes de Brownien _________________________________________________________

@jit
def List_W( T, n, N ,fonction_W ): 
  # création des listes de Brownien sauf pour MLMC 
  # n ou N doit être paire pour fonctionner avec QMC
  
  if (fonction_W==W_simple or fonction_W==W_bridge_dya  ):
    # cas W_simple ou W_bridge_dya
    L_T, W = fonction_W(T, n)
    L_W  =  np.zeros((N,len(L_T)))
    L_W[0] =  W              
    for i in range(1,N):
      L_T, W = fonction_W(T, n)
      L_W[i] =  W 
    return L_T, L_W 
  
  # cas QMC 
  L_T = np.linspace(0,T,n)
  L_W  =  np.zeros((N,n))
  nterm = N*n//2  # on genere deux liste de normales de gaussien par Boxmuller d'ou la division par 2
  
  # construcion de la liste de normales
  if (fonction_W==W_simple_QMCHM):
    U1 = stratification(nterm)
    U2 = stratification(nterm)
  if (fonction_W==W_simple_QMC):
    m = len((bin(nterm)[2:])) -1
    U = (qmc.Sobol(d=1, scramble=True).random_base2(m+2)).transpose()[0]
    U = U[U != 0]
    np.random.shuffle(U)
    U1 = U[:nterm]
    U2 = U[nterm:2*nterm]
  Z1, Z2 = Bmuller(U1, U2)
  Z = np.concatenate((Z1, Z2 ))
  Z = np.reshape(Z, (N,n))
  
  # création des Browniens 
  for i in range(N):
    for t in range(1,n):
      L_W[i, t] = L_W[i, t-1] + np.sqrt(T/n)*Z[i, t]
  return  L_T, L_W 


@jit
def L_W_anti( T, n, N, fonction_W ):
  # on double la liste de Brownien en passant à l'opposé
  L_T, L_W = List_W(T, n, N, fonction_W)

  return L_T, np.concatenate((L_W, (- L_W)))


@jit
def List_W_mlmc( T, etape, N, fonction_W): 
  # Mlmc requiert deux Browniens issus du même Brownien de finesse différentes
  gen = W_mlmc(T, etape , fonction_W, M=4)

  L_T1 = gen[0]
  W1 = gen[1]
  L_T2 = gen[2]
  W2 = gen[3]

  L_W1  =  np.zeros((N,len(L_T1)))
  L_W2  =  np.zeros((N,len(L_T2)))
  L_W1[0] =  W1   
  L_W2[0] =  W2 

  for i in range(1,N):
    gen = W_mlmc(T, etape , fonction_W, M=4)
    L_W1[i] = gen[1]
    L_W2[i] = gen[3]

  return L_T1, L_W1, L_T2, L_W2

#__________________________________Résolution numérique___________________________
@jit
def dYt_sep( i, Y, W1, W2, L_T ):
  # fonction donnant la variation de Vt et St à partir des équations  
  # i = étape 
  dW1t = W1[i]-W1[i-1]
  dW2t = W2[i]-W2[i-1]
  dt =  L_T[i]-L_T[i-1]
  if Y[0]<0:
      # quand le pas est trop grossier Vt peut devenir négatif ce qui est exclue au vu de l'équation
      print( "warning n est trop faible")
      v = 0 # on empèche la valeure négative et on le remplace par 0
  else:
      v = Y[0]
  # utilisation des équations couplées
  dVt  = k*(theta - v)*dt  +  xi*np.sqrt(v)*dW1t
  dSt  = r*Y[1]*dt  +  np.sqrt(v)*Y[1]*rho*dW1t  +  np.sqrt(1-rho**2)*dW2t
  return [dVt, dSt ]


# la méthode Runge Kuta à adapter avec le nouveau format des fonctions de génération
# def RK4(Y0, dYt, T, n):
  # dt = T/n
  # L_Y = [Y0]
  # L_T = [0]
  # for i in range(n):
  #   Yt = L_Y[-1]
  #   k1 = dYt( dt = dt, Y = Yt )
  #   k2 = dYt( dt = dt+dt/2, Y = Yt+ k1 )
  #   k3 = dYt( dt = dt+dt/2, Y = Yt+ k1 )
  #   k4 = dYt( dt = 2*dt, Y = Yt+ k1 )
  #   dY = (k1+(4/3)*k2+(4/3)*k3+(1/2)*k4)/6
  #   L_Y.append( Yt + dY )
  #   L_T.append( L_T[-1] + dt )
  # return L_T, L_Y

@jit
def L_Euler_sep( Y0, dYt_sep, L_T, L_W):
    # méthode d'Euler 
    N = len(L_W)//2 # nombre de simulation à réaliser
    n2 = len(L_T)
    L_Euler = np.empty((2*N,n2), dtype=np.float64)
    for i in range(N):
        W1 = L_W[2*i]
        W2 = L_W[2*i+1]
        L_Euler[2*i,0]   = Y0[0]
        L_Euler[2*i+1,0] = Y0[1]
        for t in range(1,n2):
            Yt = [L_Euler[2*i,t-1] , L_Euler[2*i+1,t-1]]
            dY = dYt_sep( t, Yt, W1, W2, L_T )
            L_Euler[2*i,t]   = Yt[0]+dY[0]
            L_Euler[2*i+1,t] = Yt[1]+dY[1]   
    return L_Euler
    

@jit
def L_generations(T, n, N, Y0, fonction_W, fonction_L_W ):
  # fonction qui regroupe les résultats des générations 
  L_T, L_W = fonction_L_W(T, n, 2*N, fonction_W) # w1 et W2
  L_Euler  = L_Euler_sep( Y0, dYt_sep, L_T, L_W)
  return L_Euler, L_W, L_T

@jit
def L_generations_mlmc(T, etape, N, Y0, fonction_W):
  L_T1, L_W1, L_T2, L_W2 = List_W_mlmc( T, etape, 2*N, fonction_W) # W1 et W2
  L_Euler1  = L_Euler_sep( Y0, dYt_sep, L_T1, L_W1)
  L_Euler2  = L_Euler_sep( Y0, dYt_sep, L_T2, L_W2)
  return L_Euler1, L_Euler2


#____________________________fonctions de calcul du prix de l'option___________________________
@jit
def L_PHI( T, n, N, Y0, fonction_W, fonction_L_W, K=K, r=r, l = 5):
  # donne la liste des réalistions de la variable aléatoire à l'intérieur de l'espérance
  
  L_Euler, L_W, L_T = L_generations(T, n, N, Y0, fonction_W, fonction_L_W  )
  #Affiche_Gen( N, L_Euler, L_W, L_T, brownien = True ) 
  n2 = len(L_Euler[0])
  L_S_ti = np.array([ np.mean([  L_Euler[2*i+1,t] for t in range(n2) ]) for i in range(N) ]) - K
  L_S_ti = np.array([  max(v, 0) for v in L_S_ti ])*np.exp(-r*T)
  return L_S_ti

@jit
def L_PHI_c( T, n, N, Y0, fonction_W, fonction_L_W, K=K, r=r):

  L_Euler, L_W, L_T = L_generations(T, n, N, Y0, fonction_W, fonction_L_W  )
  #Affiche_Gen( N, L_Euler, L_W, L_T, brownien = True )

  n2 = len(L_Euler[0])
  L_S_ti = np.array([ np.mean([  L_Euler[2*i+1,t] for t in range(n2) ]) for i in range(N) ]) - K
  L_S_ti = np.array([  max(v, 0) for v in L_S_ti ])*np.exp(-r*T)
  Z = np.array([ np.mean([  np.log( max(L_Euler[2*i+1,t], 1e-10)) for t in range(n2) ]) for i in range(N) ]) - K
  Z = np.exp(-r*T) * np.array([  max(v, 0) for v in Z ])*np.exp(-r*T)
  Z = np.nan_to_num(Z)
  
  return L_S_ti, Z



@jit
def L_PHI_mlmc( T, etape, Y0, N, fonction_W, K=K, r=r, l = 5):
  L_Euler1, L_Euler2 = L_generations_mlmc(T, etape, N, Y0, fonction_W)
  n2_1 = len(L_Euler1[0])
  n2_2 = len(L_Euler2[0])
  L_S_ti_1 = np.array([ np.mean([  L_Euler1[2*i+1,t] for t in range(n2_1) ]) for i in range(N) ]) - K
  L_S_ti_1 = np.array([  max(v, 0) for v in L_S_ti_1 ])*np.exp(-r*T)
  L_S_ti_2 = np.array([ np.mean([  L_Euler2[2*i+1,t] for t in range(n2_2) ]) for i in range(N) ]) - K
  L_S_ti_2 = np.array([  max(v, 0) for v in L_S_ti_2 ])*np.exp(-r*T)
  
  return L_S_ti_1, L_S_ti_2

#_____________________________estimateurs_____________________________________________

@jit
def estimateur_MC(T, n , N, Y0, fonction_W) :

  # T : maturité de l'option
  # n : nombres d'observations 
  # N : nombre de variables simulées (longueur liste de sortie)
  # Y0 : Position des paramètres à la base de la simulation du Brownien
  # fonction_W : définit la méthode de génération du Brownien (W_simple ou W_bridge)

  phi = L_PHI( T, n, N, Y0, fonction_W ,List_W )
  res = np.mean(phi)
  return(res)

@jit
def estimateur_anti(T, n , N, Y0, fonction_W) :

  # T : maturité de l'option
  # n : nombres d'observations 
  # N : nombre de variables simulées (longueur liste de sortie)
  # Y0 : Position des paramètres à la base de la simulation du Brownien
  # fonction_W : définit la méthode de génération du Brownien (W_simple ou W_bridge)

  phi = L_PHI( T, n, N, Y0, fonction_W, L_W_anti )
  res = np.mean(phi)
  return(res)

@jit
def estimateur_control(T, n , N, Y0, fonction_W) :

  # T : maturité de l'option
  # n : nombres d'observations 
  # N : nombre de variables simulées (longueur liste de sortie)
  # Y0 : Position des paramètres à la base de la simulation du Brownien
  # fonction_W : définit la méthode de génération du Brownien (W_simple ou W_bridge)

  couple = L_PHI_c( T, n, N, Y0, fonction_W, List_W)
  phi = couple[0]
  Z = couple[1]
  Z = Z - np.mean(Z)
  covariance = np.cov(phi, Z)
  beta = covariance[0][1] / covariance[1][1]
  res = np.mean(phi - beta*Z)
  return(res)


@jit
def estimateur_mlmc(T, n_max, N0, Y0, fonction_W, M=4) :

  # T : maturité de l'option
  # n_max : finesse de l'étape la plus profonde
  # On commence pour etape = 10 (donc une puissance de moins : 2**9 = 512)
  # N0 : le nombre d'estimation pour l'étape de base
  # Nk = N_(k-1) / (10**k) : décroissance selon l'article
  # Y0 : Position des paramètres à la base de la simulation du Brownien

  if (fonction_W == W_bridge_dya):

    h_max = 0

    if (n_max > 16385) :
      h_max = 6
    elif (n_max > 4097):
      h_max = 5
    elif (n_max > 1025):
      h_max = 4

    # Estimation grossière par Monte Carlo naïf
    estimateur = estimateur_MC(T, 257 , N0, Y0, fonction_W) # On commence pour n=257 (puissance 3) : sinon pas assez de points (soit l'étape 2)
    # Estimation multi-niveau par différences

    N = N0
    # On parcourt les niveaux de sommes téléscopiques
    for i in range(3,h_max):
      N = N0/(10**(1-(1/(i+1)))) # Cf décroissance empirique de l'article
      N = int(N)
      L_S_ti_1, L_S_ti_2 = L_PHI_mlmc(T, i+1, Y0, N, fonction_W , K=K, r=r, l = 5)
      estimateur += np.mean(L_S_ti_2) - np.mean(L_S_ti_1)

    return(estimateur)

  else:
    # Estimation grossière par Monte Carlo naïf
    estimateur = estimateur_MC(T, 2**9 , N0, Y0, fonction_W)

    # On déduit la profondeur h_max à partir de n_max (finesse de la couche profonde)
    h_max = int(np.log(n_max) / np.log(2))

    # Estimation multi-niveau par différences

    # On parcourt les niveaux de sommes téléscopiques
    for i in range(h_max-10):
      # Attention : on commence en finesse 10 (car en étape O, on est en finesse 9)
      N = N0/(10**(1-(1/(i+1)))) # Cf décroissance empirique de l'article
      N = int(N)
      L_S_ti_1, L_S_ti_2 = L_PHI_mlmc( T, 10+i, Y0, N, fonction_W, K=K, r=r, l = 5)
      estimateur += np.mean(L_S_ti_2) - np.mean(L_S_ti_1)

    return(estimateur)

#__________________________collecte de données_______________________________________

# attention le path est à modifier 

def newdata_df(df,n,N,nb, fonction_W, fonction_estim ):
    for i in range(nb):
        t_debut = ti.perf_counter()
        res = fonction_estim(T, n , N, Y0, fonction_W)
        # print(res)
        if np.isnan(res):
            print(" negative value ")
        else:
            t_fin = ti.perf_counter()
            newrow = {'n': n,'N': N, 'res': res, 'tps':t_fin-t_debut, 'type_W': fonction_W.__name__, 'type_estim': fonction_estim.__name__ }
            df = df.append(newrow, ignore_index = True)
            
    return df

def session_data(L_n, L_N, nb, fonction_W, fonction_estim):
    df = pd.read_csv(mainpath +"MC_data3.csv")
    for i in tqdm(range(len(L_n))):
        for j in tqdm(range(len(L_N))):
            df = newdata_df(df,L_n[i],L_N[j],nb, fonction_W, fonction_estim )
    df.to_csv(mainpath +"MC_data3.csv")
    
  
#____________________fonctions d'affichages________________________________

def plot_hist_estimateurs(nb, T, n , N, Y0, L_f_W, L_f_estim):
  for fonction_W in L_f_W:
    for fonction_estim in L_f_estim:
      Lgen = [ fonction_estim(T, n , N, Y0, fonction_W)  for k in range(nb)]
      name = " " +fonction_W.__name__+" "+ fonction_estim.__name__
      plt.hist(Lgen, bins =20, density = True, alpha = .3, label =name )
  plt.show()
  
  
def Affiche_Gen( nb, L_Euler, L_W, L_T, brownien = False):
  for k in range(nb):
    plt.plot( L_T, L_Euler[2*k] , "g", alpha = 0.6 )
    plt.plot( L_T, L_Euler[2*k+1], "b", alpha = 0.8  )

    if brownien:
      plt.plot( L_T, L_W[2*k], "r", alpha = 0.5 )
      plt.plot( L_T, L_W[2*k +1], "y", alpha = 0.5 )
     
    plt.show()


# ___________________initialisation de la database ______________________________________

# df = pd.DataFrame(
# {"n" : [],
# "N" : [],
# "res": [],
# "tps": [],
# "type_W":[],
# "type_estim":[],
#     })
# df.to_csv('../MC_data2.csv')


### ___________________Listes des paramètres pour la collecte_________________________________________

## n:
    
#L_n = np.arange(500, 5000, 500)
#L_n = np.arange(500, 5500, 500)
#L_n = np.arange(1000, 10000, 1000)
#L_n = np.arange(8000, 11000, 1000)
L_n = np.arange(1000, 8000, 2000)

## N:
    
#L_N = np.arange(50, 1000, 50)
#L_N = np.arange(50, 1000, 50)
L_N = np.arange(500, 5500, 500)

## estimateurs

#L_f_W = [ W_bridge_dya, W_simple ] 
L_f_W = [W_simple_QMC, W_simple_QMCHM]

#L_f_estim = [ estimateur_control, estimateur_mlmc ]
L_f_estim = [ estimateur_MC, estimateur_anti]

#____________Main collecte de données____________________________________________

# t_debut =ti.perf_counter()
# for f_W in L_f_W:
#     for f_estim in L_f_estim:     
#         session_data(L_n, L_N, 6, f_W , f_estim)
# t_fin = ti.perf_counter()
# print(t_fin-t_debut)

#df2 = pd.read_csv(mainpath +"MC_data3.csv")

#__________________tests unitaires___________________________

#print( estimateur_MC(5,1000, 100, Y0, W_simple_QMCHM))
#print( estimateur_control(5,1000, 100, Y0, W_simple_QMC))
#print( estimateur_anti(5, 1000, 100, Y0, W_bridge_dya))
#print(estimateur_control(5, 5000 , 1000, Y0, W_simple_QMC))
#print(estimateur_mlmc(5, 5000 , 1000, Y0, W_bridge_dya))