
data(iris)
flores=data.frame(iris)

summary(flores)

normaliza <-function(x) {(x-min(x))/(max(x)-min(x))}

flores_normalizadas=as.data.frame(lapply(iris[,c(1,2,3,4)], normaliza))

summary(flores_normalizadas)

set.seed(73)

amostra <- sample(1:nrow(flores),0.8 * nrow(flores))

iris_train=flores_normalizadas[amostra,]

iris_test=flores_normalizadas[-amostra,]

#Como ao longo do modelo criamos uma matriz para ter uma melhor visualização
#da classificação dos nossos dados, nossa percentagem de acerto é dada pelo dessa matriz dividido pela soma total

precisao <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

#Aqui rodamos a 10-fold Cross validation para o parâmetro k do nosso modelo num intervalo de 1:57 com passo 4; estes valores 
# bem elevados foram propositais para visualizar o que acontece para uma vasta gama de Ks


Erro_cv=list()
Erro_fold=list()

for(j in (seq(1,57,by=4))){
    for(i in 1:10){
        amostracv=as.integer(nrow(flores)*(i-1)/10+1):as.integer(nrow(flores)/10*i)
        treino=flores_normalizadas[-amostracv,]
        teste=flores_normalizadas[amostracv,]
        iris_target_category <- flores[-amostracv,5]
        iris_test_category <- flores[amostracv,5]
        modelo=knn(treino,teste,cl=iris_target_category,k=j)
        tabela <- table(modelo,iris_test_category)  #aqui criamos uma maneira de visualizar como nosso modelo classificou os dados
        Erro_fold=c(Erro_fold,precisao(tabela))       
    }
    Erro_fold=unlist(Erro_fold)
    Erro_cv=c(Erro_cv,mean(Erro_fold))
    Erro_fold=list()              
}

Erro_cv=unlist(Erro_cv)

Erro_cv

k=seq(1,57,by=4)
plot(k,Erro_cv,type='l',main='Precisão da validação cruzada em função de K',ylab='1-Erro Cross Validation',xlab='Número de vizinhos próximos')

Erro_cv[5]

iris_target_category <- iris[amostra,5]
iris_test_category <- iris[-amostra,5]

modelo=knn(iris_train,iris_test,cl=iris_target_category,k=17)

tabela=table(modelo,iris_test_category)

tabela

precisao(tabela)
