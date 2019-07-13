# Trabalho de ML

#### *Construindo um modelo preditivo de classificação de tumores de mama*



O relatório do trabalho será feito neste arquivo Markdown. Além disso, o documento está organizado como respostas às perguntas do [arquivo](https://www.dropbox.com/s/pnc02xjuotf7q0y/Trabalho.pdf?dl=0) com as instruções.



#### *Antepasti*

Antes de iniciar a resolução, cabe indicar os materiais de consulta. Durante a execução do trabalho, senti que estava um pouco despreparado para tarefas intermediárias e práticas como: obtenção dos dados, limpeza e uso da sintaxe do Pandas (biblioteca de Python). Assim, busquei materiais de consulta **com abordagens práticas**, entre eles, destaco:

- Repositório do professor Renato Souza da EMAp, em especial, o *case* da IRIS [link](https://github.com/rsouza/FGV_Intro_DS)
- Vídeo IRIS [link](https://www.youtube.com/watch?v=hd1W4CyPX58)
- *Playlist* do Youtuber sentdex com vários exemplos, modelo preditivo de ações e, inclusive, o de câncer de mama que será discutido aqui [link](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)
- Stackoverflow: perguntas diversas feitas por outros usuários e perguntas que eu mesmo fiz [link](https://stackoverflow.com/questions/56718635/is-the-interpretation-of-the-data-visualization-bellow-which-uses-python-librar)



## Parte 1: Introdução



#### 1.1 - Descreva a base em conjunto com sua *pergunta de pesquisa*



A base de dados usada foi extraída do repositório da **UC Irvine** chamado **Machine Learning Repository** [link](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/).

Os dados descrevem atributos de células cancerígenas na região da mama, sendo que o principal ponto é a indicação de uma classificação entre tumores malignos e benignos.

A pergunta de pesquisa que se pretende responder é: *em qual nível de acurácia um modelo preditivo consegue classificar um tumor, na região da mama, como benigno ou maligo?*

#### 1.2 - Apresente a base de dados e a dependência de suas variáveis



Os principais aspectos da base são:



-  A amostra é formada por `699` elementos, em `241` casos o tumor era *benigno* e em `458` casos o tumor era *maligno*.

- Ao todo são 11 colunas/atributos na base de dados, sendo que uma delas é apenas um identificador (ID) e a última delas apresenta a classificação como Benigno ou Maligno. Para tumores benignos, o número 2 é usado. Para tumores malignos, o número 4 é usado.

  De acordo com a documentação da base de dados, essas são as variáveis:

  ```tex
  
     #  Attribute                     Domain
     -- -----------------------------------------
     1. Sample code number            id number
     2. Clump Thickness               1 - 10
     3. Uniformity of Cell Size       1 - 10
     4. Uniformity of Cell Shape      1 - 10
     5. Marginal Adhesion             1 - 10
     6. Single Epithelial Cell Size   1 - 10
     7. Bare Nuclei                   1 - 10
     8. Bland Chromatin               1 - 10
     9. Normal Nucleoli               1 - 10
    10. Mitoses                       1 - 10
    11. Class:                        (2 for benign, 4 for malignant)
  
  ```

- A única variável dependente é `Class`. As outras  colunas como `Normal Nucleoli` e etc são modeladas como variáveis independentes, com uma única exceção: a coluna `Sample code number` não é uma variável no modelo. Ela serve apenas como apoio para a manipulação dos dados.

  

  Por fim, há de ser ressaltado que o problema a ser tratado neste trabalho é a construção de um modelo preditivo de classificação de tumores como malignos ou benignos **usando o método de KNN**.



#### 1.3 Apresente estatísticas descritivas da base



Por meio de um script simples em Python foram geradas estatísticas descritivas relevantes para compreensão do contexto:

```python
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

for i in df:
    print (i)
    print (df[str(i)].describe())
    print (" ")

```



Seria possível um `dt.describe()`, mas a leitura fica melhor no formato abaixo, com um `print` para cada variável.

As estatísticas descritivas mais diretas da base de dados são:

```
clump_thickness
count    699.000000
mean       4.417740
std        2.815741
min        1.000000
25%        2.000000
50%        4.000000
75%        6.000000
max       10.000000
Name: clump_thickness, dtype: float64

unif_cel_size
count    699.000000
mean       3.134478
std        3.051459
min        1.000000
25%        1.000000
50%        1.000000
75%        5.000000
max       10.000000
Name: unif_cel_size, dtype: float64

unif_cel_shape
count    699.000000
mean       3.207439
std        2.971913
min        1.000000
25%        1.000000
50%        1.000000
75%        5.000000
max       10.000000
Name: unif_cel_shape, dtype: float64

marg_adhesion
count    699.000000
mean       2.806867
std        2.855379
min        1.000000
25%        1.000000
50%        1.000000
75%        4.000000
max       10.000000
Name: marg_adhesion, dtype: float64

single_epith_cell_size
count    699.000000
mean       3.216023
std        2.214300
min        1.000000
25%        2.000000
50%        2.000000
75%        4.000000
max       10.000000
Name: single_epith_cell_size, dtype: float64

bare_nuclei
count     699
unique     11
top         1
freq      402
Name: bare_nuclei, dtype: object

bland_chrom
count    699.000000
mean       3.437768
std        2.438364
min        1.000000
25%        2.000000
50%        3.000000
75%        5.000000
max       10.000000
Name: bland_chrom, dtype: float64

norm_nucleoli
count    699.000000
mean       2.866953
std        3.053634
min        1.000000
25%        1.000000
50%        1.000000
75%        4.000000
max       10.000000
Name: norm_nucleoli, dtype: float64

mitoses
count    699.000000
mean       1.589413
std        1.715078
min        1.000000
25%        1.000000
50%        1.000000
75%        1.000000
max       10.000000
Name: mitoses, dtype: float64

```



Como todo o trabalho gira em torno da classificação de tumores como benignos ou malignos, cabe apresentar a diferença nessas estatísticas entre tumores malignos e benignos.



```
bland_chrom                                                
            count      mean       std  min  25%  50%  75%   max   
class                                                             
2           458.0  2.100437  1.080339  1.0  1.0  2.0  3.0   7.0   
4           241.0  5.979253  2.273852  1.0  4.0  7.0  7.0  10.0   


clump_thickness                                                    
                count      mean       std  min  25%  50%   75%   max  count   
class                                                                         
2               458.0  2.956332  1.674318  1.0  1.0  3.0   4.0   8.0  458.0   
4               241.0  7.195021  2.428849  1.0  5.0  8.0  10.0  10.0  241.0   


marg_adhesion                                               
              max         count      mean       std  min  25%  50%  75%   max   
class                                                                           
2      13454352.0         458.0  1.364629  0.996830  1.0  1.0  1.0  1.0  10.0   
4       1371026.0         241.0  5.547718  3.210465  1.0  3.0  5.0  8.0  10.0   


mitoses                                                 
        count      mean       std  min  25%  50%  75%   max         count   
class                                                                       
2       458.0  1.063319  0.501995  1.0  1.0  1.0  1.0   8.0         458.0   
4       241.0  2.589212  2.557939  1.0  1.0  1.0  3.0  10.0         241.0   


norm_nucleoli                                           
           mean       std  min  25%  50%   75%   max                  count   
class                                                                         
2      1.290393  1.058856  1.0  1.0  1.0   1.0   9.0                  458.0   
4      5.863071  3.350672  1.0  3.0  6.0  10.0  10.0                  241.0   


single_epith_cell_size                                           
           mean       std  min  25%  50%  75%   max          count      mean   
class                                                                          
2      2.120087  0.917130  1.0  2.0  2.0  2.0  10.0          458.0  1.443231   
4      5.298755  2.451606  1.0  3.0  5.0  6.0  10.0          241.0  6.560166   


unif_cel_shape                                                  
            std  min  25%  50%  75%   max         count      mean       std   
class                                                                         
2      0.997836  1.0  1.0  1.0  1.0   8.0         458.0  1.325328  0.907694   
4      2.562045  1.0  4.0  6.0  9.0  10.0         241.0  6.572614  2.719512   


unif_cel_size                            
       min  25%  50%   75%   max  
class                             
2      1.0  1.0  1.0   1.0   9.0  
4      1.0  4.0  6.0  10.0  10.0  
```

 

Analisando a tabela acima, é possível perceber que, nos valores máximos e nos valores mínimos, os tumores malignos e benignos não são muito diferentes. No caso de valores mínimos, são praticamente idênticos, seja no tamanho uniforme da célular ou na mitose celular, tanto tumores benignos como malignos recebem nota 1 como valor **mínimo** na escala de 1 a 10.

Entre os valores máximos em cada uma das variáveis é possível encontrar algumas diferenças entre tumores malignos e beginos. Por exemplo, o máximo *Bland Chromatin* em tumores benignos recebete nota 7, enquanto que o máximo na mesma váriavel (*Bland Chromatin*) em tumores malignos recebe nota 10. Apesar desta diferença, os valores **máximos** entre os dois tumores nas dez variáveis consideradas são, de certa forma, **homogêneos**, sendo a maior diferença identificada de três unidades.

**A diferença chave entre os dois tipos de tumores não está nos valores máximos ou mínimos de cada variável, mas sim nos valores que estão no quartil superior, isto é, no 75º percentil.** Nessa faixa dos dados, as estatísticas são bem heterogêneas entre tumores benignos e malignos. Por exemplo, na variável *Uniform Cell Size*, a nota atribuída para o quartil superior é de 1 em 10 para tumores beginos. Em tumores malignos, por sua vez, neste percentil 75º a nota atribuída é de 10. Como a escala usada pelos pesquisadores que montaram a base de dados vai de 1 a 10, essa é maior diferença possível: nove unidades.

Em outras variáveis como  *Marg adhesion* e *Norm Nucleoli* essa assimetria também aconteceu. Isso sugere que casos nessa faixa de percentil sejam mais fáceis de serem classificados  do que casos que estão no percentil 1º ou 100º.

#### 1.4 Apresente outra duas análises ou visualizações interessantes



A documentação da base de dados está disponibillizada no arquivo`breast-cancer-wisconsin.names`. No documento, indica-se a seguinte distribuição de tumores benignos e malignos na amostra :

* Benign: `458 ` (`65.5%`)
* Malignant: `241` (`34.5%`)

Assim, decidi plotar os dados por meio do script:



```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
#print(df.head())  

#df["bare_nuclei"].hist()
df["class"].hist()

#x = df["class"].hist()
#plt.xticks(x, ['a','c'])

plt.title('Maligno versus Benigno')
plt.ylabel('Frequência')
plt.figtext(.75, .01, "4 = Tumor Maligno")
plt.figtext(.15, .01, "2 = Benigno")
x = [2,4]
plt.xticks(x)
y = [0,50,100,150,200,250,300,350,400,450,500,550,600]
plt.yticks(y)
plt.show()

```



O código retorna a seguinte imagem:

![alt text](https://github.com/pdelfino/learning-from-data/blob/master/ml-project/visu-central-info.png)



Apesar de simples, a imagem acima ilustra a informação crucial dos dados.

Duas visualizações mais interessantes e menos óbvias  serão mostradas abaixo.  Entretando, cabe contar brevemente a origem por trás delas.

Uma das sugestões da comunidade para trabalhos práticos de *data science* e de *machine learning* é o envolvimento de um especialista do domínio no processo, isto é, recomenda-se que o projeto seja discutido com um profissional que entende o **contexto** de onde os dados foram retirados. Idealmente, o próprio desenvolvedor de machine learning é conhecedor do domínio envolvido. 

No meu caso, tenho conhecimentos apenas de ensino médio sobre biologia. Assim, busquei conversar com um médico, que apesar de  cirurgião (e não oncologista),  deu uma contribuição. Ao saber da proposta de "advinhar o futuro" (modelo de predição) do trabalho, o profissional de saúde indicou que o tamanho do tumor seria provavelmente um bom indicativo para descobrir se ele é maligno ou benigno. 

Com esse comentário, decidi validar a hipótese. Fiz um histograma em relação ao tamanho dos tumores e uma separação por cores. Para isso, usei a variável  `Uniformity of Cell Size`.  Essa variável vai de 1 a 10, sendo 10 o tamanho máximo de uma célula. De fato, a hipótese/intuição  do ''especialista'' no domínio estava correta. Como o histograma abaixo indica,  tumores malignos tendem a ser maiores que benignos:

![alt text](https://github.com/pdelfino/learning-from-data/blob/master/ml-project/histogram-cell-size.png)



![alt text](https://github.com/pdelfino/learning-from-data/blob/master/ml-project/size-maligno-benigno.png)



O código usado para a primeira visualização é:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df.head())  
#df['unif_cel_size'].hist()
df.groupby('class')['unif_cel_size'].plot(kind='hist', color="black")

plt.xlabel("Uniform Cell Size (0-10)", fontdict=None, labelpad=None, )
plt.ylabel("Frequência do número de casos", fontdict=None, labelpad=None, )
x = [1,2,3,4,5,6,7,8,9,10]
plt.xticks(x)
plt.show()

```

E para a segunda:



```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df.head())  
#df['unif_cel_size'].hist()
df.groupby('class')['unif_cel_size'].plot(kind='hist')

plt.xlabel("Uniform Cell Size (0-10)", fontdict=None, labelpad=None, )
plt.ylabel("Frequência do número de casos", fontdict=None, labelpad=None, )

plt.gca().legend(("2-benigno","4-maligno"))
plt.show()

```



#### 1.5 Proporção de treino e de teste

Dividi os meus dados aleatoriamente na proporção de `70-30`, sendo que `70%` foram usados para treinamento e `30%` para teste.



# Parte 2: Metodologia 



#### 2.1 Indicação das variáveis dependentes e independentes que serão usadas

Como dito acima, as variáveis independentes são:

- `Clump Thickness   `
- `Uniformity of Cell Size  `

- `Uniformity of Cell Shape `
- `Marginal Adhesion  `
- `Single Epithelial Cell Size `
- `Bare Nuclei `
- `Bland Chromatin `
- `Normal Nucleoli  `
- `Mitoses` 

A única variável dependente será:

- `Class`        

Como dito anteriormente, a variável `Sample code number` serve apenas como ferramenta de manipulação dos dados.



#### 2.2 Identificação do hiperparâmetro



O método usado será o KNN. Desse modo, existe um único hiperparâmetro a ser definido: o k-número de vizinhos que serão  considerados.

Seguindo as recomendações de boas práticas que percebi no conteúdo prático citado acima, a recomendação é de usar a raiz quadrada do tamanho da amostra, isto é: 

k = \sqrt{n} 

No caso,  a amostra é `n = 699`. Assim, a raiz aproximada é `k = 26`.



#### 2.3 Utilize k-fold cross validation para selecionar pelo menos 15 valores do hiperparâmetro

O objetivo do *cross validation* é escolher um `k` ótimo.  Assim, irei testar o modelo de predição para todos os valores de `k` entre `1` e `50`.

O script usado foi:



```python
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# no arquivo de dados são 11 variáveis, 9 independetes, 1 dependente e 1 é só primary-key
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# para dados faltantes
df.replace('?',-99999, inplace=True)

# tira a coluna de id pq ela não é nem variável dependente e nem independente
df.drop(['id'], 1, inplace=True)

# as features são tudo menos a coluna de classe, a id já foi excluída
X = np.array(df.drop(['class'], 1))

# o Y na minha supervised learning é a class
y = np.array(df['class'])

# separar nos conjuntos de treino e de teste, para depois descobrir a acurácia
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

k_values = list(range(1,51))


for i in k_values:

    # defini o hiperparâmetro como k=i, 26 está dentro do intervalo de iteração 
    clf = neighbors.KNeighborsClassifier(i)

    # inserir no objeto o treinamento
    clf.fit(X_train, y_train)

    # fazer o teste para saber a acurácia
    accuracy = clf.score(X_test, y_test)
    
    print("acurácia para o hiperparâmatro k=", i, ": ", accuracy)
    #print("erro no teste: ", 1-accuracy)
```



O output é:



```
acurácia para o hiperparâmatro k= 1 :  0.9523809523809523
acurácia para o hiperparâmatro k= 2 :  0.9428571428571428
acurácia para o hiperparâmatro k= 3 :  0.9476190476190476
acurácia para o hiperparâmatro k= 4 :  0.9476190476190476
acurácia para o hiperparâmatro k= 5 :  0.9523809523809523
acurácia para o hiperparâmatro k= 6 :  0.9523809523809523
acurácia para o hiperparâmatro k= 7 :  0.9476190476190476
acurácia para o hiperparâmatro k= 8 :  0.9523809523809523
acurácia para o hiperparâmatro k= 9 :  0.9571428571428572
acurácia para o hiperparâmatro k= 10 :  0.9571428571428572
acurácia para o hiperparâmatro k= 11 :  0.9571428571428572
acurácia para o hiperparâmatro k= 12 :  0.9571428571428572
acurácia para o hiperparâmatro k= 13 :  0.9571428571428572
acurácia para o hiperparâmatro k= 14 :  0.9523809523809523
acurácia para o hiperparâmatro k= 15 :  0.9523809523809523
acurácia para o hiperparâmatro k= 16 :  0.9523809523809523
acurácia para o hiperparâmatro k= 17 :  0.9523809523809523
acurácia para o hiperparâmatro k= 18 :  0.9523809523809523
acurácia para o hiperparâmatro k= 19 :  0.9523809523809523
acurácia para o hiperparâmatro k= 20 :  0.9571428571428572
acurácia para o hiperparâmatro k= 21 :  0.9571428571428572
acurácia para o hiperparâmatro k= 22 :  0.9571428571428572
acurácia para o hiperparâmatro k= 23 :  0.9571428571428572
acurácia para o hiperparâmatro k= 24 :  0.9571428571428572
acurácia para o hiperparâmatro k= 25 :  0.9571428571428572
acurácia para o hiperparâmatro k= 26 :  0.9571428571428572
acurácia para o hiperparâmatro k= 27 :  0.9571428571428572
acurácia para o hiperparâmatro k= 28 :  0.9571428571428572
acurácia para o hiperparâmatro k= 29 :  0.9571428571428572
acurácia para o hiperparâmatro k= 30 :  0.9571428571428572
acurácia para o hiperparâmatro k= 31 :  0.9571428571428572
acurácia para o hiperparâmatro k= 32 :  0.9523809523809523
acurácia para o hiperparâmatro k= 33 :  0.9523809523809523
acurácia para o hiperparâmatro k= 34 :  0.9523809523809523
acurácia para o hiperparâmatro k= 35 :  0.9523809523809523
acurácia para o hiperparâmatro k= 36 :  0.9523809523809523
acurácia para o hiperparâmatro k= 37 :  0.9523809523809523
acurácia para o hiperparâmatro k= 38 :  0.9523809523809523
acurácia para o hiperparâmatro k= 39 :  0.9523809523809523
acurácia para o hiperparâmatro k= 40 :  0.9523809523809523
acurácia para o hiperparâmatro k= 41 :  0.9523809523809523
acurácia para o hiperparâmatro k= 42 :  0.9523809523809523
acurácia para o hiperparâmatro k= 43 :  0.9523809523809523
acurácia para o hiperparâmatro k= 44 :  0.9523809523809523
acurácia para o hiperparâmatro k= 45 :  0.9523809523809523
acurácia para o hiperparâmatro k= 46 :  0.9523809523809523
acurácia para o hiperparâmatro k= 47 :  0.9523809523809523
acurácia para o hiperparâmatro k= 48 :  0.9523809523809523
acurácia para o hiperparâmatro k= 49 :  0.9571428571428572
acurácia para o hiperparâmatro k= 50 :  0.9523809523809523
```



Ordenando os valores de forma crescente, percebemos que `k=26` valor sugerido pela regra de bolso utilizada é um dos resultados que obteve **a maior acurácia no teste** - empatado com outros valores ,diga-se de passagem: 

```
k=	2	0.9428571428571428
k=	3	0.9476190476190476
k=	4	0.9476190476190476
k=	7	0.9476190476190476
k=	1	0.9523809523809523
k=	5	0.9523809523809523
k=	6	0.9523809523809523
k=	8	0.9523809523809523
k=	14	0.9523809523809523
k=	15	0.9523809523809523
k=	16	0.9523809523809523
k=	17	0.9523809523809523
k=	18	0.9523809523809523
k=	19	0.9523809523809523
k=	32	0.9523809523809523
k=	33	0.9523809523809523
k=	34	0.9523809523809523
k=	35	0.9523809523809523
k=	36	0.9523809523809523
k=	37	0.9523809523809523
k=	38	0.9523809523809523
k=	39	0.9523809523809523
k=	40	0.9523809523809523
k=	41	0.9523809523809523
k=	42	0.9523809523809523
k=	43	0.9523809523809523
k=	44	0.9523809523809523
k=	45	0.9523809523809523
k=	46	0.9523809523809523
k=	47	0.9523809523809523
k=	48	0.9523809523809523
k=	50	0.9523809523809523
k=	9	0.9571428571428572
k=	10	0.9571428571428572
k=	11	0.9571428571428572
k=	12	0.9571428571428572
k=	13	0.9571428571428572
k=	20	0.9571428571428572
k=	21	0.9571428571428572
k=	22	0.9571428571428572
k=	23	0.9571428571428572
k=	24	0.9571428571428572
k=	25	0.9571428571428572

k=	26	0.9571428571428572

k=	27	0.9571428571428572
k=	28	0.9571428571428572
k=	29	0.9571428571428572
k=	30	0.9571428571428572
k=	31	0.9571428571428572
k=	49	0.9571428571428572

```

 

#### 2.4 Construa o gráfico com erro de validação cruzada

Usando o script abaixo, selecionei `15` valores de `k` entre `1` e `489`.

A amostra vai até `699`. Entretanto, tive que limitar para um valor inferior, caso contrário, na etapa seguinte, quando eu fosse rodar a acurácia do modelo preditivo para cada um desses valores, eu poderia ter `k` maior do que o `n`. Portanto, limitei os valores possíveis de `k` entre `1` e `489`, sendo que `489`  é `70%` de `699`, arredondando para baixo.

```python
import random

lista_valores_hiperparametros = random.sample(range(1,489),15)

print (lista_valores_hiperparametros)
lista_valores_hiperparametros.append(26)

lista_valores_hiperparametros.sort()
print (lista_valores_hiperparametros)

```



Ao invés de construir um gráfico usando o erro, eu usei o ''complementar'', o nível de acurácia. 

Além disso, fiz questão de inserir o valor k=26, sugerido pela ''regra de bolso''. É possível constatar que conforme o valor do hiperparâmetro se distancia do recomendado, pior é a acurácia do modelo. Assim, quanto maior o valor do hiperparâmetro, mais expressivo é o erro.

Portanto, a intuição do uso da raiz quadrada na regra de bolso é: ela serve para "achatar" o valor do hiperparâmetro em função do tamanho da amostra.

A imagem abaixo ilustra:

![alt text](https://github.com/pdelfino/learning-from-data/blob/master/ml-project/grafico-k-variando.png)



O código usado é:

```python
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import random
import matplotlib.pyplot as plt

# no arquivo de dados são 11 variáveis, 10 independetes e 1 dependente
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?',-99999, inplace=True)

df.drop(['id'], 1, inplace=True)

# as features são tudo menos a coluna de classe
X = np.array(df.drop(['class'], 1))

y = np.array(df['class'])

# separar nos conjuntos de treino e de teste, para depois descobrir a acurácia
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

lista_valores_hiperparametros = random.sample(range(1,489),15)
print (lista_valores_hiperparametros)
lista_valores_hiperparametros.append(26)

lista_valores_hiperparametros.sort()
print (lista_valores_hiperparametros)

accuracy_list = []
for i in lista_valores_hiperparametros:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
    clf = neighbors.KNeighborsClassifier(i)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    accuracy_list.append(accuracy)

print (accuracy_list)

plt.plot(lista_valores_hiperparametros, accuracy_list , 'ro')
plt.axis([1,489,0,1])
plt.xlabel('Valores para o hiperparâmetro K')
plt.ylabel('Nível de Acurácia')
plt.title('Gráfico com o erro para cada um dos parâmetros')
plt.show()
```



#### 2.5 Plote a fronteira de decisão para o hiperparâmetro escolhido

Não é simples plotar a fronteira de decisão tendo em  vista que são 10 variáveis envolvidas. Nos exemplos vistos em sala, tínhamos duas variáveis e um plano de duas dimensões. 

Diante desse impasse, resolvi fazer uma simplificação. Vou plotar a fronteira de decisão utilizando 2 variáveis independentes em cada eixo e a variável dependete (que indica a classe  do tumor como benigno ou maligno), será usada para indicar a cor. As duas variáveis independentes utilizadas foram: *uniformity cell size* (eixo horizontal - x) e *marginal adhesion* (eixo vertical - y).

Obviamente, a fronteira de decisão plotada abaixo não reflete a acurácia do modelo que de fato foi usado em outras questões.

```python
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

style.use('fivethirtyeight')

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?',-99999, inplace=True)

unif_cel_size = df['unif_cel_size']
#print (unif_cel_size)

marg_adhesion = df['marg_adhesion']
#print (marg_adhesion)

classif = df['class']
#print (classif)

vals = []

for i in range(0,len(classif)):
    vals.append([unif_cel_size[i],marg_adhesion[i],classif[i]])

# criar dicionário
# iterar sobre vals, se terminar em 2, adicionar à parte com 2, se terminar em 4, adicionar à parte maligno

for i in vals:
    if i[2]==2:

        plt.scatter(i[0],i[1],s=50,color='b')
    elif i[2]==4:
        plt.scatter(i[0],i[1],s=50,color='r')
plt.xlabel('uniform cell')
plt.ylabel('marginal adhesion')
plt.title('Fronteira de decisão para 2 variáveis')
plt.show()
```





A imagem é:



![alt text](https://github.com/pdelfino/learning-from-data/blob/master/ml-project/fronteira-decisao.png)



Como é possível ver acima, em virtude da mistura dos dados não é possível traçar uma linha que os separe.

#### 2.6 Apresente os erros dentro da amostra, de validação cruzada e de teste

Para `k=26`

Acurácia: `0.9571428571428572`

Erro de teste: `0.042857142857142816`

Erro de treinamento: `0.03271983640081799`



Código com comentário linha por linha:

```python
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# no arquivo de dados são 11 variáveis, 10 independetes e 1 dependente
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# para dados faltantes
df.replace('?',-99999, inplace=True)

# tira a coluna de id pq ela não é nem variável dependente e nem independente
df.drop(['id'], 1, inplace=True)

# as features são tudo menos a coluna de classe, a id já foi excluída
X = np.array(df.drop(['class'], 1))

# o Y na minha supervised learning é a class
y = np.array(df['class'])

# separar nos conjuntos de treino e de teste, para depois descobrir a acurácia
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

# defini o hiperparâmetro como k=26 já que n=699 como tamanho da amostra
clf = neighbors.KNeighborsClassifier(26)

# inserir no objeto o treinamento
clf.fit(X_train, y_train)

# fazer o teste para saber a acurácia
accuracy = clf.score(X_test, y_test)

print("acurácia: ", accuracy)
print("erro no teste: ", 1-accuracy)
print ("erro no treinamento: ", 1-clf.score(X_train,y_train))
```



# Parte 3: Conclusão

## Geral



O resultado foi satisfatório. O modelo preditivo se comportou bem e teve alto nível de acurácia. Com o valor de hiperparâmetro  `k=26` , que segue a recomendação da área, o nível de acurácia gira em torno de `95%`.

Como visto acima, o valor do hiperparâmetro é peça chave para o nível de acurácia, já que valores de `k` muito grandes reduzem a acurácia do modelo.



# Extra



No momento da entrega da A2, coloquei como *future work* a elaboração de outros modelos preditivos para o mesmo problema. A ideia era ver como a acurácia se comportaria diante de um modelo mais simples e de um modelo mais sofisticado. Como sugerido nas aulas do Professor Eduardo Mendes, ao avaliar a qualidade de um modelo preditivo sempre devemos ter um *lower bound* de referência.

Como modelo mais sofisticado que o KNN, será usado o *Support Vector Machine (SVM)*. Como modelo mais simples que o KNN, será usada uma regressão lógistica simples.



## Support Vector Machine



O código para SVM foi feito utilizando o código do knn de base. Assim, tudo que foi descrito acima em relação às variáveis independentes, dependentes, proporção de treino/teste foi mantido. Houve a alteração basicamente de uma linha:



```python
 clf = svm.SVC(kernel='rbf')

```



O `sklearn`, no caso do método *Support Vector Machine*, contempla diversos parâmetros. No caso do método KNN,  a escolha era basicamente em torno do valor de k.

Por simplificação, todos os valores foram mantidos no *default*, sendo definidos implicitamente. A única exceção foi a explicitação do `kernel`, ponto que foi um dos mais comentados em sala durante o curso.



Entre as diversas opções de kernel como: `linear`, `poly`, `sigmoid` - optei por escolher `rbf`, indicando a radial basis function, que, por sinal, foi um dos assuntos escolhidos pelo Professor para serem implementados durante o curso. O código completo é:

```python
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

# no arquivo de dados são 11 variáveis, 9 independetes, 1 dependente e 1 primary key
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# para dados faltantes
df.replace('?',-99999, inplace=True)

# tira a coluna de id pq ela não é nem variável dependente e nem independente, só primary key
df.drop(['id'], 1, inplace=True)

# as features são tudo menos a coluna de classe, a id já foi excluída
X = np.array(df.drop(['class'], 1))

# o Y na minha supervised learning é a class
y = np.array(df['class'])

# separar nos conjuntos de treino e de teste, para depois descobrir a acurácia
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

# poderia ser 'linear', 'poly', 'sigmoid'. Pelo que vimos em sala, usar 'rbf' 
clf = svm.SVC(kernel='rbf')

# inserir no objeto o treinamento
clf.fit(X_train, y_train)

# fazer o teste para saber a acurácia
accuracy = clf.score(X_test, y_test)

print("accuracy: ", accuracy)
print("test error: ", 1-accuracy)
print ("trainning error: ", 1-clf.score(X_train,y_train))

'''
#exemplo pra prever se a célula é benigna ou maligna
example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print("predição: ",prediction)
'''
```





O output foi:



```
accuracy:  0.9666666666666667
test error:  0.033333333333333326
trainning error:  0.002044989775051076
```



Como é possível ver, **a acurácia do modelo SVM foi levemente superior a do modelo KNN.** Portanto, a maior sofisticação no método implicou, nesse caso, em maior acurácia, ainda que a melhora tenha sido relativamente pequena.



## Regressão Logística 



O modelo de regressão logística está sendo usado aqui como alternativa mais simples ao KNN.

Novamente, o código, os dados, as variáveis dependetes e independentes são as idênticas ao que foi usado para aplicação dos métodos de KNN e de SVM. Assim como na adaptação para aplicação do método SVM, houve a alteração de apenas uma linha:



```python
clf = linear_model.LogisticRegression()
```

Para gerar o mínimo de interferência possível, foram mantidos todos os parâmetros por *default*. O código completo é:



```python
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm, linear_model
import pandas as pd

# no arquivo de dados são 11 variáveis, 9 independetes, 1 dependente e 1 primary key
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# para dados faltantes
df.replace('?',-99999, inplace=True)

# tira a coluna de id pq ela não é nem variável dependente e nem independente, só primary key
df.drop(['id'], 1, inplace=True)

# as features são tudo menos a coluna de classe, a id já foi excluída
X = np.array(df.drop(['class'], 1))

# o Y na minha supervised learning é a class
y = np.array(df['class'])

# separar nos conjuntos de treino e de teste, para depois descobrir a acurácia
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

############################################################## 
clf = linear_model.LogisticRegression()

# inserir no objeto o treinamento
clf.fit(X_train, y_train)

# fazer o teste para saber a acurácia
accuracy = clf.score(X_test, y_test)

print("accuracy: ", accuracy)
print("test error: ", 1-accuracy)
print ("trainning error: ", 1-clf.score(X_train,y_train))

```





 Assim, o output foi:



```
accuracy:  0.9380952380952381
test error:  0.06190476190476191
trainning error:  0.024539877300613466

```



### Conclusão

Conforme o que foi visto, a sofisticação de métodos implicou uma melhora dos resultados, sendo que a maior acurácia foi atribuída à *Support Vector Machine*.





