
# coding: utf-8

# In[32]:


import math
import numpy as np
import random


# In[39]:


def euclideandistance(x,y):
    resultado=0
    if type(x)!=list:
        return(x-y)**2
    else:
        for i in range(len(x)):
            resultado+=(x[i]-y[i])**2
        return(math.sqrt(resultado))


# In[6]:


def fi(Z):
    return(math.exp(-.5*(Z**2)))


# In[45]:


def fi_RBFnetwork(D,x,r):
    X=[a[0] for a in D]
    resultado=np.zeros(len(X))
    for i in range(len(X)):
        resultado[i]==fi(abs(elemento-x)/r)
    return resultado


# In[41]:


#Fitting the data pg 29

def W_RBFnetwork(D,k,r):
    X=[a[0] for a in D]
    Y=[a[1] for a in D]
    
    'Determinamos os K centros!'
    C=random.sample(X,k)
    Centro=Centros(C,X,K,100000000) #iniciamos com um erro muito alto para garantir que o próximo passo tenha um erro menor
        
    Z=np.zeros((len(X),k+1))
    for i in range(len(X)):
        for j in range(K+1):
            if j==0:
                Z[i][j]==1
            else:
                Z[i][j]==fi((abs(X[i]-Centro[j]))/r)
    W=numpy.linalg.inv(Z*numpy.matrix.transpose(Z))*numpy.matrix.transpose(Z)*Y
    return W


# In[40]:


#Lloyd algorithm


def Centros(C,X,K,erro):
    Centrositer=[]
    S=[] #lista composta pelas listas dos elementos pertencentes a um determinado centro
    for a in range(k):
        S.append([])
    for dado in X:
        p=-1
        l=0
        for n in range(K):
            if p<0:
                p=euclideandistance(C0[n],dado)
                l=n
            elif p<euclideandistance(C0[n],dado):
                l=n
        S[l].append(dado)
    for elemento in S:
        Centrositer.append(sum(elemento)/len(elemento))
    
    erroiter=funcao_erro(S,Centrositer)
    
    if erro<= erroiter:
        return C
    
    else:
        Centros(Centrositer,X,K,erroiter)


# In[43]:


def funcao_erro(S,Centro):
    erro=0
    for i in range(len(Centro)):
        for elemento in S[i]:
            erro+=euclideandistance(elemento,Centro[i])**2
    return erro       


# In[47]:


def RBFnetwork(D,x,K,r):
    learning=np.matrix.transpose(W_RBFnetwork(D,K,r))*fi_RBFnetwork(D,x,r)
    return learning


# In[51]:


def CrossValidation(D,r):
    '''
    Estamos validando os dados usando a validação cruzada "todos menos um"
    
    '''
    erro=[]
    for elemento in D:
        A=D.copy
        A.remove(elemento)
        X=[a[0] for a in A]
        Y=[a[1] for a in A]
        erro.append(erroRBF(X,Y,elemento))
    return sum(erro)/len(D)

def escolhe_r(R,D):
    ''' 
    R é o conjunto de valores de r que iremos testar    
    '''
    H=[]
    for r in R:
        H.append(CrossValidation(D,r))
    return R[H.index(min(H))]


# In[52]:


def erroRBF(X,Y,p):
    '''
    X e Y são o conjunto de valores
    p é o ponto sobre o qual será calculado o erro
    '''
    N=0
    D=0
    for i in range(len(X)):
        alfa=math.exp(-.5*abs(X[i]-p[0]))
        N+=alfa*y[i]
        D+=alfa
    return abs(N/D-p[1])

