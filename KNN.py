
# coding: utf-8

# In[336]:


import math
import numpy as np
import matplotlib.pyplot as plt
import random


# Função que retorna a distancia Euclidiana entre dois pontos ou vetores

# In[337]:


def distancia_Euclidiana(x,y):
    resultado=0
    if type(y)!=list:
        resultado=(x-y)**2
    else:
        for i in range(len(x)):
            resultado+=(x[i]-y[i])**2
    return math.sqrt(resultado)


# Função que retorna o vetor formado pelas distancias ponto a ponto

# In[338]:


def vetordistancia(D,x):
    resultado=[]
    for i in range(len(D)):
        resultado.append(distancia_Euclidiana(D[i][0],x))
    return resultado            


# Função que retorna os K valores mais próximos

# In[356]:


def Kmin(v,K):
    ValMin=[]
    l=v.copy()
    for i in range(K):
        ValMin.append(min(l))
        l.remove(min(l))
    indices=[]
    for i in ValMin:
        indices.append(v.index(i))
    return indices


# Função KNN

# In[340]:


'''
D: Dados na forma de uma matriz de N vetores v
v: vetores de duas dimensões onde o primeiro parâmetro é o conjunto de dados que sabemos a respeito do input
e o segundo parâmetro é o resultado de tal input
x: vetor a ser aproximado
'''
def KNN(D,x,K):
    V=vetordistancia(D,x)
    pontos=Kmin(V,K)
    aproximacao=0
    for i in pontos:
        aproximacao+=D[i][1]/K
    return aproximacao


# Testando a aproximação para um conjunto de pontos aleatórios com ruido da função objetivo Sen

# In[341]:


l=list(np.linspace(-math.pi,math.pi,40))
dados=random.sample(l,21)
dados.sort()


# In[348]:


funcaoobjetivo=[math.sin(a) for a in dados]


# In[357]:


valcomruido=[math.sin(a)+(random.random()-.5)/3 for a in dados]
plt.plot(dados,valcomruido)
plt.plot(dados,funcaoobjetivo)


# In[360]:


D=[(a,math.sin(a)+(random.random()-.5)/3) for a in dados]
valoresKNN=[KNN(D,a,1) for a in dados]
plt.plot(dados,valoresKNN)
plt.plot(dados,funcaoobjetivo)


# In[352]:


D=[(a,math.sin(a)+(random.random()-.5)/3) for a in dados]
valoresKNN=[KNN(D,a,3) for a in dados]
plt.plot(dados,valoresKNN)
plt.plot(dados,funcaoobjetivo)


# In[355]:


D=[(a,math.sin(a)+(random.random()-.5)/3) for a in dados]
valoresKNN=[KNN(D,a,11) for a in dados]
plt.plot(dados,valoresKNN)
plt.plot(dados,funcaoobjetivo)

