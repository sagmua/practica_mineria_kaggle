#Librerías a usar:
library(dplyr)
library(Hmisc)
library(mice)
library(ggplot2)
library(caret)
library(lattice)
library(VIM)
library(Amelia)
library(partykit)
library(RWeka)
library(tree)
require(robCompositions)
library(NoiseFiltersR)
library(outliers)
require(mvoutlier)
library(mlbench)
library(corrplot)
library(FSelector)
library(unbalanced)
library(ROSE)
library(Boruta)

#Funciones auxiliares:
setwd("~/GitHub/practica_mineria_kaggle")
source("funcionesAux.R")


############ EJECUTAR EL SCRIPT COMPLETO PARA OBTENER EL MODELO FINAL ##########

#lectura de datos:
setwd("~/GitHub/practica_mineria_kaggle")
train = read.csv("train.csv", na.string=c(" ", "NA", "?"))
test = read.csv("test.csv", na.string=c(" ", "NA", "?"))
#Pasamos a formato TIBBLE:
train = as_tibble(train)
train$C = as.factor(train$C)
test = as_tibble(test)
dim(train)
dim(test)


#Resumen de datos
summary(train)

#Eliminar duplicados
train = unique(train)
dim(train)


#Proporción con gráfica del balanceo de clase
datos = train
datos$C = as.factor(datos$C)
proporcion = group_by(datos,C) %>% summarise(propo = round((nc = (n() * 100)/dim(datos)[1]),digits = 1))
proporcion

df <- data.frame(clase=proporcion$C,
                 proporcion=proporcion$propo)

p<-ggplot(df, aes(x=clase, y=proporcion, fill=clase)) +
  geom_bar(stat="identity")+theme_minimal()
p

#Patrón de missing values
patron <- mice::md.pattern(x=train[,-51], plot=TRUE)

completas <- mice::ncc(train)
incompletas <- mice::nic(train)


completas
incompletas

#Se visualiza un grafico que nos indica la forma en que se distribuyen los 
# datos perdidos
# se genera el grafico de distribucion de datos perdidos . Solo 
# se consideran las variables con datos perdidos
plot = aggr(train , col=c("blue","red") , numbers=TRUE, sortVars=TRUE,labels=names(train), cex.axis=.5, gap=1, ylab=c("Grafico de datos perdidos","Patron"))

#Otro gráfico más de distribución de NAs
cat("Datos completos: ",completas, " e incompletos: ",incompletas,"\n")
missmap(train[1:300,1:20], col= c("red","steelBlue"))


# se realiza la imputacion con mice
# 5 batch (series) diferentes de imputaciones. El método cart es el de imputación que se utiliza
imputados <- mice::mice(train, m=5, meth="cart", parallel = "multicore")

# se completa el conjunto de datos con las imputaciones
datosImputados <- mice::complete(imputados)


# se determina el numero de instancias sin datos perdidos y con datos
# perdidos en la parte ya limpia
completos <- mice::ncc(datosImputados)
incompletos <- mice::nic(datosImputados)
cat("Datos completos: ",completos, " e incompletos: ",incompletos,"\n")


#Imputación con Amelia
imputados.amelia <- Amelia::amelia(train,m=5,parallel="multicore",noms="C")
incompletos.amelia <- mice::nic(imputados.amelia$imputations$imp5)
completos.amelia <- mice::ncc(imputados.amelia$imputations$imp5)
cat("COMPLETOS = ", completos.amelia, " INCOMPLETOS = ",incompletos.amelia)
datosImputadosAmelia <- imputados.amelia$imputations$imp5



# Imputación con impKNNa
imputados <- robCompositions::impKNNa(train, primitive=TRUE)

# Ahora puede visualizarse alguna informacion sobre la forma
# en que se hizo la imputacion. El segundo argumento indica el
# tipo de grafico a obtener
plot(imputados, which=2)

# El conjunto de datos completo puede accederse de la siguiente forma
datosImputadosRobCompositions =  imputados$xImp

# se determina el numero de instancias sin datos perdidos y con datos
# perdidos. A observar la comodidad de uso de las funciones ncc e nic
completos <- mice::ncc(datosImputadosRobCompositions)
incompletos <- mice::nic(datosImputadosRobCompositions)
cat("Datos completos: ",completos, " e incompletos: ",incompletos,"\n")

# Filtrado de ruido con IPF

# se inicializa la semilla aleatoria para reproducir los resultados
set.seed(1)

datos = as.data.frame(datosImputadosAmelia)

datos$C = as.factor(datos$C)
# se aplica el algoritmo 
# El s = 2 establece el criterio de parada el cual indica que el algoritmo para
# después de encontrarse sin eliminar instancias tras dos iteraciones
out <- IPF(C~., data = datos, s = 2)

# se muestran los indices de las instancias con ruido
#summary(out, explicit = TRUE)

# el conjunto de datos sin ruido se obtiene de la siguiente forma
out$cleanData

# tambien podriamos obtenerlo de forma directa eliminando los
# indices de las instancias consideradas como ruidosas
datosSinRuido <- datos[setdiff(1:nrow(datos),out$remIdx),]

datosSinRuido = as.data.frame(datosSinRuido)
str(datosSinRuido)


#Proporción de clases después de aplicar el filtro de ruido
proporcion = group_by(datosSinRuido,C) %>% summarise(propo = round((nc = (n() * 100)/dim(datosSinRuido)[1]),digits = 1))
proporcion

df <- data.frame(clase=proporcion$C,
                 proporcion=proporcion$propo)

p<-ggplot(df, aes(x=clase, y=proporcion, fill=clase)) +
  geom_bar(stat="identity")+theme_minimal()
p

# Filtrado de ruido mediante EF

# se inicializa la semilla aleatoria para reproducir los resultados
set.seed(1)

datos = as.data.frame(datosImputadosAmelia)

datos$C = as.factor(datos$C)
# se aplica el algoritmo 
out = EF(C~., data = datos, consensus = FALSE)

# se muestran los indices de las instancias con ruido
#summary(out, explicit = TRUE)

# el conjunto de datos sin ruido se obtiene de la siguiente forma
out$cleanData

# tambien podriamos obtenerlo de forma directa eliminando los
# indices de las instancias consideradas como ruidosas
datosSinRuidoEF <- datos[setdiff(1:nrow(datos),out$remIdx),]

datosSinRuidoEF = as.data.frame(datosSinRuidoEF)
str(datosSinRuidoEF)


# Filtrado de ruido mediante CVCF

# se inicializa la semilla aleatoria para reproducir los resultados
set.seed(1)

datos = as.data.frame(datosImputadosAmelia)

datos$C = as.factor(datos$C)
# se aplica el algoritmo 
out = CVCF(C~., data = datos, consensus = FALSE)

# se muestran los indices de las instancias con ruido
#summary(out, explicit = TRUE)

# el conjunto de datos sin ruido se obtiene de la siguiente forma
out$cleanData

# tambien podriamos obtenerlo de forma directa eliminando los
# indices de las instancias consideradas como ruidosas
datosSinRuidoCVCF <- datos[setdiff(1:nrow(datos),out$remIdx),]

datosSinRuidoCVCF = as.data.frame(datosSinRuidoCVCF)
str(datosSinRuidoCVCF)


# Boxplot dónde podemos ver la presencia de anomalías en el conjunto de datos
boxplot(datosSinRuido[,-51])



# Probar a balancear el conjunto de datos usando oversampling con SMOTE
salidaSmote = ubSMOTE(datosSinRuido[,-51],datosSinRuido$C)

datosSmote = cbind(salidaSmote$X, C = salidaSmote$Y)

dim(datosSmote)

proporcion = group_by(datosSmote,C) %>% summarise(propo = round((nc = (n() * 100)/dim(datosSmote)[1]),digits = 1))
proporcion

df <- data.frame(clase=proporcion$C,
                 proporcion=proporcion$propo)

p<-ggplot(df, aes(x=clase, y=proporcion, fill=clase)) +
  geom_bar(stat="identity")+theme_minimal()
p


# Probar a balancear el conjunto de datos usando oversampling con ROS
set.sed(1)
data.balanced.ou <- ovun.sample(C~., data=datosSinRuido, p=0.5, seed=1, method="over")$data

dim(data.balanced.ou)

proporcion = group_by(data.balanced.ou,C) %>% summarise(propo = round((nc = (n() * 100)/dim(data.balanced.ou)[1]),digits = 1))
proporcion

df <- data.frame(clase=proporcion$C,
                 proporcion=proporcion$propo)

p<-ggplot(df, aes(x=clase, y=proporcion, fill=clase)) +
  geom_bar(stat="identity")+theme_minimal()
p


# Probar a balancear el conjunto de datos usando undersampling con TomekLink
set.seed(2)
#Primera ejecución
salidaTomek = ubTomek(datosSinRuido[,-51], datosSinRuido$C)

datosTomek = cbind(salidaTomek$X, C = salidaTomek$Y)

dim(datosTomek)

proporcion = group_by(datosTomek,C) %>% summarise(propo = round((nc = (n() * 100)/dim(datosTomek)[1]),digits = 1))
proporcion

df <- data.frame(clase=proporcion$C,
                 proporcion=proporcion$propo)

p<-ggplot(df, aes(x=clase, y=proporcion, fill=clase)) +
  geom_bar(stat="identity")+theme_minimal()
p

set.seed(2)
#Segunda Ejecución
salidaTomek = ubTomek(datosTomek[,-51], datosTomek$C)

datosTomek = cbind(salidaTomek$X, C = salidaTomek$Y)

dim(datosTomek)

proporcion = group_by(datosTomek,C) %>% summarise(propo = round((nc = (n() * 100)/dim(datosTomek)[1]),digits = 1))
proporcion

proporcion1 = group_by(datosSinRuido,C) %>% summarise(propo = (nc = (n())))
proporcion1
proporcion1 = group_by(datosTomek,C) %>% summarise(propo = (nc = (n())))
proporcion1


df <- data.frame(clase=proporcion$C,
                 proporcion=proporcion$propo)

p<-ggplot(df, aes(x=clase, y=proporcion, fill=clase)) +
  geom_bar(stat="identity")+theme_minimal()
p

#Análisis de Correlación

corrMatrix = cor(na.omit(datosSinRuido[,-51]))
corrplot::corrplot(corrMatrix, type = "upper", order="FPC", tl.col = "black", tl.srt = 45)

set.seed(3)
#filtrado de variables:
#el umbral establecido es 0.99999
altamenteCorreladas = caret::findCorrelation(corrMatrix, cutoff=0.99999)

datosSinCorrelacion = datosSinRuido [, -altamenteCorreladas]
datosSinCorrelacion = as_tibble(datosSinCorrelacion)

dim(datosSinCorrelacion)

#Matriz después de quitar las variables correladas anteriores

corrMatrix = cor(na.omit(datosSinCorrelacion[,-38]))

corrplot::corrplot(corrMatrix, type = "upper", order="FPC", tl.col = "black", tl.srt = 45)



# Selección de variables: Filter

# Chi-Cuadrado


# se calculan los pesos de los atributos: la medida devuelta
# indica el nivel de dependencia de cada atributo frente a la
# variable clase
weights <- FSelector::chi.squared(C~.,datosSinCorrelacion)
print(weights)

# se seleccionan los 5 mejores
subset <- FSelector::cutoff.k(weights,40)

# se muestran los seleccionados
f <- as.simple.formula(subset,"C")
print(f)
#Class ~ X17 + X1 + X16 + X2 + X6
#datosFChiSquared = datosSinCorrelacion[,-c(31)]
#datosFChiSquared


# Correlation FSelector

set.seed(2)
# se calculan los pesos mediante correlacion lineal, La variable
# clase se llama medv
d = datosTomek
d$C = as.integer(datosTomek$C)
weights <- FSelector::linear.correlation(C~., d)

# se muestran los pesos
print(weights)

# se seleccionan los tres mejores
subset <- FSelector::cutoff.k(weights,38)
f1 <- as.simple.formula(subset,"C")
print(f1)
barplot(weights$attr_importance, names.arg = rownames(weights), las=2)

# se determinan los pesos mediante rank.correlation
weights <- FSelector::rank.correlation(C~.,d)

# se muestran los pesos
print(weights)

# se seleccionan los mejores
subset <- FSelector::cutoff.k(weights,33)
f2 <- as.simple.formula(subset,"C")
print(f2)
barplot(weights$attr_importance, names.arg = rownames(weights), las=2)

# Entropy

# se obtienen las medidas mediante ganancia de informacion
weights <- FSelector::information.gain(C~., datosSinRuido)

# se muestran los pesos y se seleccionan los mejores
print(weights)
subset <- FSelector::cutoff.k(weights,4)
f1 <- as.simple.formula(subset,"C")
print(f1)

# igual, pero con ganancia de informacion
weights <- FSelector::gain.ratio(C~., datosSinRuido)
print(weights)
subset <- FSelector::cutoff.k(weights,45)
f2 <- as.simple.formula(subset,"C")
print(f2)
# e igual con symmetrical.uncertainty

weights <- FSelector::symmetrical.uncertainty(C~., datosSinCorrelacion)
print(weights)
subset <- FSelector::cutoff.k(weights,35)
f3 <- as.simple.formula(subset,"C")
print(f3)



# One R

# se calculan los pesos
weights <- FSelector::oneR(C~.,datosSinRuido)



# Selección de variables: Embebed

# Random Forest
# se muestran los resultados
print(weights)
subset <- FSelector::cutoff.k(weights,50)
f <- as.simple.formula(subset,"C")
print(f)


# Seleccion de variables: Wrapper

#Paquete Boruta

#aprende el modelo
Bor.son <- Boruta(C~.,data=datosTomek,doTrace=2)

# muestra los resultados
print(Bor.son)

# se ven los resultados de decision de cada variable
print(Bor.son$finalDecision)

# imprime las estadisticas
stats <- attStats(Bor.son)
print(stats)

# se muestran los resultados en forma grafica
plot(Bor.son)

# muestra un grafico de los resultado: los valores en
# rojo estan relacionados con las variables confirmadas
# mientras que los verdes con variables descartadas
plot(normHits~meanImp,col=stats$decision,data=stats)

# aplicacion del metodo de seleccion
Bor.ozo <- Boruta(C~.,data=datosTomek,doTrace=2)

cat("Random forest sobre todos los atributos\n")
model1 <- randomForest(C~.,data=datosTomek)
print(model1)

cat("Random forest unicamente sobre atributos confirmados\n")
model2 <- randomForest(datosTomek[,getSelectedAttributes(Bor.ozo)],datosTomek$C)
print(model2)

# se muestra un grafico con los resultados de la seleccion
plot(Bor.ozo)


####### Ejemplo evaluación tree

set.seed (2)

datos = datosSinRuido
# Construyo el arbol sobre el conjunto de entrenamiento
tree.train = tree(as.factor(C)~.,datos)
#tree.train

summary(tree.train)

plot(tree.train)
text(tree.train, pretty=0)

#tree.train

# Aplico el arbol sobre el conjunto de test
tree.pred1 =predict(tree.train, test ,type ="class")

#Aplicamos cv para intentar podar el árbol búscando el mejor número de nodos hoja.


cv.train = cv.tree(tree.train ,FUN=prune.misclass )
names(cv.train )
cv.train


# Ahora podamos el arbol con prune.misclass estableciendo en best el mejor valor para el número de hojas obtenido por cv.tree, en este caso 4
prune.train =prune.misclass (tree.train ,best = 4)
par(mfrow =c(1,1))
plot(prune.train)
text(prune.train ,pretty =0)

###Comprobar acierto en el train
tree.pred.train =predict (prune.train , datos ,type="class")
#table(tree.pred.train ,datos$C)
precision(tree.pred.train ,datos$C)


# Comprobar acierto en el test
tree.pred=predict (prune.train , test ,type="class")
setwd("~/GitHub/practica_mineria_kaggle")
generarEnvio(tree.pred, path = "./decisionTrees/")

######## Ejemplo evaluación con J48

set.seed (2)

datos = datosTomek
## Comprobar acierto en el train
modelC4.5 = J48(as.factor(C)~., data=datos)
cv_resul = evaluate_Weka_classifier(modelC4.5,numFolds=10, seed = 2)
cv_resul

#Dibujar árbol
plot(modelC4.5)

#modelC4.5 = J48(C~., datos) 
precision(modelC4.5$predictions ,datos$C)

#Comprobar acierto en el test
modelC4.5.pred = predict(modelC4.5, test)
setwd("~/GitHub/practica_mineria_kaggle")
generarEnvio(modelC4.5.pred, path = "./decisionTrees/")


######## Ejemplo evaluación con J48

set.seed (2)

datos = datosTomek
## Comprobar acierto en el train
modelLMT = LMT(C~., data=datos)
cv_resul = evaluate_Weka_classifier(modelLMT,numFolds=10, seed = 2)
cv_resul

#modelC4.5 = LMT(f1, datos) 
precision(modelLMT$predictions ,datos$C)

#Dibujar árbol
plot(modelLMT)

#Comprobar acierto en el test
modelLMT.pred = predict(modelLMT, test)
setwd("~/GitHub/practica_mineria_kaggle")
generarEnvio(modelLMT.pred, path = "./decisionTrees/")
