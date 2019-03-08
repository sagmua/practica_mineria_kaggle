library(dplyr)
library(Hmisc)
library(caret)
library(mice)
library(VIM)
library(class)
library(outliers)
library(ggplot2)
library(mvoutlier)
library(FSelector)
library(corrplot)
library(NoiseFiltersR)
library(lattice)
library(unbalanced)

source("../funcionesAux.R")

#Función para generar el modelo


generarModelo <- function(train){
  control = trainControl(method="cv", number=5)
  model = caret::train(train[,-ncol(train)], train$C, method="knn",
                       metric="Accuracy", trControl=control,
                       tuneGrid = data.frame(.k = seq(1,21,2)))
  model
}

#Lectura de los datos

datos <- read.csv("../train.csv",na.strings=c(" ","NA","?"))
test <- read.csv("../test.csv",na.strings=c(" ","NA","?"))


#Elminación de instancias con más de un 5% de valores perdidos


porcentajes <- apply(datos,1, function(x) sum(is.na(x))) / ncol(datos) * 100
eliminar <- porcentajes>5
which(eliminar == TRUE)


#Imputación de valores perdidos con Mice y pmm

completos <- mice::ncc(datos)
incompletos <- mice::nic(datos)
cat("COMPLETOS = ", completos, " INCOMPLETOS = ",incompletos)
imputados <- mice::mice(datos,m=5,meth="pmm")

datos.imputados <- mice::complete(imputados)

#Imputación de valores perdidos con Amelia

imputados.amelia <- Amelia::amelia(datos,m=1,parallel="multicore",noms="C")
incompletos.amelia <- mice::nic(imputados.amelia$imputations$imp1)
completos.amelia <- mice::ncc(imputados.amelia$imputations$imp1)
cat("COMPLETOS = ", completos.amelia, " INCOMPLETOS = ",incompletos.amelia)
datos <- imputados.amelia$imputations$imp1


#Imputación de valores perdidos con Knn

imputados <- robCompositions::impKNNa(datos, primitive=TRUE)
datos <- imputados$xImp
completos <- mice::ncc(datos)
incompletos <- mice::nic(datos)
cat("Datos completos: ",completos, " e incompletos: ",incompletos,"\n")
datos <- as.data.frame(datos)

#Normalización de datos

datosPrep <- caret::preProcess(datos[,-ncol(datos)],method = c("center","scale"))
datos[,-ncol(datos)] <- predict(datosPrep,datos[,-ncol(datos)])
test <- predict(datosPrep,test)

#Eliminación de ruido

datos$C <- as.factor(datos$C)
out <- IPF(C~., data = datos,s = 3)
summary(out, explicit =TRUE)
out$cleanData
datosSinRuido <- datos[setdiff(1:nrow(datos),out$remIdx),]
cat("Se han eliminado ", dim(datos)[1]- dim(datosSinRuido)[1], "instancias")
datos <- datosSinRuido

#Desbalanceo de los datos

#Undersampling Tomek link

balanced <- ubTomek(datos[,-ncol(datos)],datos$C)
datos.balanceados <- balanced$X
datos.balanceados$C <- balanced$Y

#Undersampling metodo percPos

balanced <- ubUnder(X=datos[,-ncol(datos)], Y= datos$C, perc = 40, method = "percPos")
datos.undersampling <- balanced$X
datos.undersampling$C <- balanced$Y



#Oversampling con smote

balanced <- ubSMOTE(X=datos[,-ncol(datos)], Y= datos$C)
datos.oversampling <- balanced$X
datos.oversampling$C <- balanced$Y


#Eliminación de outliers basada en IQR. 

#Se ponen a NA los valores correspondientes a los outliers y luego se eliminan dichas instancias.

datosOutliersNA <- apply(datos[,1:(ncol(datos)-1)],2,function(c){
  cuartil.primero <- quantile(c,probs=0.25,na.rm = TRUE)
  cuartil.tercero <- quantile(c,probs=0.75,na.rm = TRUE)
  iqr <- cuartil.tercero - cuartil.primero
  extremo.inferior.outlier.normal <- cuartil.primero - 1.5*iqr
  extremo.superior.outlier.normal  <- cuartil.tercero + 1.5*iqr
  
  c[c<=extremo.inferior.outlier.normal] <- NA
  c[c>=extremo.superior.outlier.normal] <- NA
  c
})

#Selección de caracteristicas

#correlacion

pesosCorrelacion <- FSelector::linear.correlation(C~., datos)
subset.correlation <- FSelector::cutoff.k(pesosCorrelacion,20)
datos1.correlation <- datos[,subset.correlation]
datos1.correlation$C <- datos$C
test1.correlation <- test[,subset.correlation]

#Chi-squared

pesosChi.squared <- FSelector::chi.squared(C~., datos)
subset.chisquared <- FSelector::cutoff.k(pesosChi.squared,20)
datos2.chisquared <- datos[,subset.chisquared]
datos2.chisquared$C <- datos$C
test2.chisquared <- test[,subset.chisquared]

#Entropy based

pesosEntropy.based <- FSelector::information.gain(C~., datos)

subset.entropybased<- FSelector::cutoff.k(pesosEntropy.based,20)
datos3.entropybased <- datos[,subset.entropybased]
datos3.entropybased$C <- datos$C
test3.entropybased <- test[,subset.entropybased]

#One R

pesosOneR <- FSelector::oneR(C~., datos)

subset.oneR<- FSelector::cutoff.k(pesosOneR,20)
datos4.oneR <- datos[,subset.oneR]
datos4.oneR$C <- datos$C
test4.oneR <- test[,subset.oneR]

#Random Forest

pesosRandomForest <- FSelector::random.forest.importance(C~.,datos, importance.type=1)

subset.RandomForest<- FSelector::cutoff.k(pesosRandomForest,20)
datos5.RandomForest <- datos[,subset.RandomForest]
datos5.RandomForest$C <- datos$C
test5.RandomForest <- test[,subset.RandomForest]

#Para generar el modelo basta con hacer un preprocesamiento usando los métodos que queramos de los descritos
#Anteriormente y llamar a la función generar modelo como en el siguiente comando:



datos$C <- as.factor(datos$C)
model <- generarModelo(datos)
model

test_pred <- predict(model,test)
test_pred

generarEnvio(y=test_pred,file="envio.csv")


