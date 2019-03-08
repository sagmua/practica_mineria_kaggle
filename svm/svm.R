## ---- include=FALSE------------------------------------------------------
#Intalacion de las librerias
#Ejecutaremos esto solo para instalar los paquetes necesarios que no tengamos previamente en el sistema. Para ello solo descomentamos la linea del paquete que nos falte y la ejecutamos.
#Esta libreria sera la principal para aplicar SVM al conjunto de datos que tenemos, con las posteriores realizaremos el preprocesamiento de los datos
# install.packages("e1071")
# install.packages("dplyr")
# install.packages("Hmisc")
# install.packages("arules")
# install.packages("mice")
# install.packages("mice")
# install.packages("Amelia")
# install.packages("PerformanceAnalytics")
# install.packages("caret")
# install.packages("corrplot")
# install.packages("VIM")
# install.packages("RWeka")
# install.packages("NoiseFiltersR")
# install.packages("FSelector")
# install.packages("RKEEL")
# install.packages("unbalanced")
# install.packages("DMwR")

#Carga de las librerias
library(e1071)
#library(dplyr)
library(Hmisc)
#library(arules)
#library(mice)
library(Amelia)
library(PerformanceAnalytics)
#library(caret)
library(corrplot)
library(VIM)
library (NoiseFiltersR)
library(FSelector)


#Funcion auxiliar
source("../funcionesAux.R")

## ------------------------------------------------------------------------
datos = read.csv("../train.csv", na.string=c(" ", "NA", "?"))
test = read.csv("../test.csv", na.string=c(" ", "NA", "?"))

## ------------------------------------------------------------------------
nrow(datos)

## ------------------------------------------------------------------------
summary(datos)

## ------------------------------------------------------------------------
nrow(test)

## ------------------------------------------------------------------------
summary(test)

## ------------------------------------------------------------------------
datos_duplicados = duplicated(datos)

## ------------------------------------------------------------------------
length(which(datos_duplicados))
datos = datos[!datos_duplicados,]

proporcion_perdidos <- apply(datos,1, function(x) sum(is.na(x))) / ncol(datos) * 100

mean(proporcion_perdidos)

## ------------------------------------------------------------------------
valores_perdidos = apply(datos,1, function(x) length(which(is.na(x))))
nrow(datos[which(valores_perdidos>1),])

## ------------------------------------------------------------------------
(nrow(datos[which(valores_perdidos==1),])/nrow(datos))*100

## ------------------------------------------------------------------------
datos_mice = mice::md.pattern(x = datos)
porcentaje_correcto = mice::ncc(datos)
porcentaje_perdidos = mice::nic(datos)

## ------------------------------------------------------------------------
imputados.amelia <- Amelia::amelia(datos,m=5,parallel="multicore",noms="C")
incompletos.amelia <- mice::nic(imputados.amelia$imputations$imp1)
completos.amelia <- mice::ncc(imputados.amelia$imputations$imp1)
cat("COMPLETOS = ", completos.amelia, " INCOMPLETOS = ",incompletos.amelia)
datos <- imputados.amelia$imputations$imp1

## ------------------------------------------------------------------------
datos_en_raw = datos
datos_con_alta_correlacion = caret :: findCorrelation(cor(datos) , cutoff =0.99)
datos = datos[,-highlyCorrelated]

#Eliminamos los datos que no aparecen correlacionados
corrplot(cor(datos))

weights = FSelector::linear.correlation(C~., datos)
barplot(weights$attr_importance, names.arg = rownames(weights), las=2)


## ------------------------------------------------------------------------
subset = FSelector :: cutoff.k(weights ,6)
#subset = cutoff.k(weights ,6)
#subset = cutoff.k(weights ,9)
conjunto = as.simple.formula(subset ,"C")
weights = rank.correlation(C~.,datos)
barplot(weights$attr_importance, names.arg = rownames(weights), las=2)
datos$C = as.factor(datos$C)
datos_en_raw$C = as.factor(datos_en_raw$C)
print(conjunto)



## ------------------------------------------------------------------------
set.seed (1)
out = IPF(C~., data = datos , nfolds=5, s = 3)
summary(out, explicit =TRUE)
identical(out$cleanData , datos[ setdiff (1:nrow(datos) ,out$remIdx) ,])
datos = out$cleanData

datos_en_raw_out = IPF(C~., data = datos_en_raw , nfolds=5, s = 3)
datos_en_raw = datos_en_raw_out$cleanData

## ------------------------------------------------------------------------
datos_anomalos <- outliers::outlier(datos[,-ncol(datos)])
resOutliers1 <- mvoutlier::uni.plot(datos[1:6])


## ------------------------------------------------------------------------
print(conjunto)

## ------------------------------------------------------------------------
print(datos)

## ------------------------------------------------------------------------
print(test)

## ------------------------------------------------------------------------
#x <- subset(test, select=-C)
#y <- C

svm_model <- svm(conjunto, data = datos)
#svm_model <- svm(conjunto, data = test)

summary(svm_model)


pred <- predict(svm_model,datos)

system.time(pred <- predict(svm_model,datos))

#print(pred)

#svm_tune <- tune(svm, conjunto, data = datos, kernel="radial", ranges=list(cost=1, gamma=c(.2,.5,1,2)))
svm_tune <- tune(svm, conjunto, data = datos, kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

print(svm_tune)

#svm_model_after_tune <- svm(  svm, conjunto, data = datos, kernel="radial", cost=1, gamma=0.5 )
svm_model_after_tune <- svm(  conjunto, data=datos, kernel="radial", cost=1, gamma=0.0689526 )
summary(svm_model_after_tune)

pred <- predict(svm_model_after_tune,datos)
system.time(predict(svm_model_after_tune,datos))



## ------------------------------------------------------------------------
#precision(predictionLabels = prediccion, testLabels = datos$C)

prediccion = predict(svm_model_after_tune, test)

#generarEnvio(pred, file = "testEnvSVM10.csv")
generarEnvio(prediccion, file = "EnvioSVM18.csv")


