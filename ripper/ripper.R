## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

## ---- include=FALSE------------------------------------------------------
#Librerías a usar y funciones útiles:
library(dplyr)
library(Hmisc)
library(mice)
library(Amelia)
library(PerformanceAnalytics)
library(caret)
library(corrplot)
library(VIM)
library(RWeka)
library ( NoiseFiltersR )
library(FSelector)
library(RKEEL)
library(unbalanced)
library(DMwR)



##FUNCIONES AUXILIARES:

source("../funcionesAux.R")

## ---- include=FALSE------------------------------------------------------
#lectura de datos:
datos = read.csv("../train.csv", na.string=c(" ", "NA", "?"))
test = read.csv("../test.csv", na.string=c(" ", "NA", "?"))

## ------------------------------------------------------------------------
duplicados = duplicated(datos)
length(which(duplicados))

## ---- include=FALSE------------------------------------------------------
datos = datos[!duplicados,]

## ---- include=FALSE------------------------------------------------------
anyNA(datos)

#numero de valores perdidos por cada instancia:
numero_na = apply(datos,1, function(x) length(which(is.na(x))))


## ------------------------------------------------------------------------
nrow(datos[which(numero_na>1),])

## ------------------------------------------------------------------------
nrow(datos[which(numero_na==1),])


## ---- echo=FALSE---------------------------------------------------------
plot <- VIM::aggr(datos,col=c('blue','red'),numbers=TRUE,sortVars=TRUE,labels=names(data),
                  cex.axis=.5,gap=1,ylab=c("Gráfico de datos perdidos","Patron"))

## ---- include=FALSE------------------------------------------------------
patron = mice::md.pattern(x = datos)

## ---- echo = FALSE-------------------------------------------------------
completas = mice::ncc(datos)
incompletas = mice::nic(datos)

df <- data.frame(
  group = c("Inompletas", "Completas"),
  value = c(incompletas, completas)
  )
bar <- ggplot(df, aes( x = "", y = value, fill = group)) + geom_bar(width = 1, stat = "identity") 
pie <- bar + coord_polar("y", start=0) + geom_text(aes(x=1, y = cumsum(value) - value*0.5, label=round((value/nrow(datos)*100),digits = 2)))
pie

## ---- results="hide", echo=TRUE, warning=FALSE---------------------------
#imputados = mice::mice(datos, m=5, meth="pmm")

## ------------------------------------------------------------------------
#imputados$method[imputados$method == "pmm"]

## ------------------------------------------------------------------------
#IMPUTACION CON MICE_
#var_rest = names(imputados$method[imputados$method == "pmm"])
#datos = complete( imputados)
#datos = datos[,append(var_rest, "C")]
#summary(datos)

#IMPUTACION CON AMELIA:
imputados.amelia <- Amelia::amelia(datos,m=1,parallel="multicore",noms="C")
incompletos.amelia <- mice::nic(imputados.amelia$imputations$imp1)
completos.amelia <- mice::ncc(imputados.amelia$imputations$imp1)
cat("COMPLETOS = ", completos.amelia, " INCOMPLETOS = ",incompletos.amelia)
datos <- imputados.amelia$imputations$imp1

## ------------------------------------------------------------------------
datos_sin_seleccion = datos
# se encuentran aquellas variables que presentan valores de correlacion
# por encima del valor umbral
highlyCorrelated = caret :: findCorrelation(cor(datos[,-ncol(datos)]) , cutoff =0.99999)
datos = datos[,-highlyCorrelated]
corrplot(cor(datos))

## ------------------------------------------------------------------------
#Calculamos los pesos mediante la correlación linear con nuestra
# variable clase 'C'
weights = FSelector::linear.correlation(C~., datos)
barplot(weights$attr_importance, names.arg = rownames(weights), las=2)


## ------------------------------------------------------------------------
subset = cutoff.k(weights ,4)
f = as.simple.formula(subset ,"C")
weights = rank.correlation(C~.,datos)
barplot(weights$attr_importance, names.arg = rownames(weights), las=2)

subset = FSelector :: cutoff.k(weights ,9)
f = as.simple.formula(subset ,"C")
print(f)

## ------------------------------------------------------------------------
#datos = datos[,!names(datos) %in% c("X25", "X44")]
#datos [,-which(names(datos) == "C")] = scale(datos[,- which(names(datos) == "C")], center = TRUE, scale = TRUE)
datos$C = as.factor(datos$C)
datos_sin_seleccion$C = as.factor(datos_sin_seleccion$C)

## ---- include=FALSE------------------------------------------------------
set.seed (1)
out = IPF(C~., data = datos , nfolds=5, s = 3)
summary(out, explicit =TRUE)
identical(out$cleanData , datos[ setdiff (1:nrow(datos) ,out$remIdx) ,])
datos = out$cleanData

datos_sin_seleccion_out = IPF(C~., data = datos_sin_seleccion , nfolds=5, s = 3)
datos_sin_seleccion = datos_sin_seleccion_out$cleanData


## ------------------------------------------------------------------------
anomalos <- outliers::outlier(datos[,-ncol(datos)])
print(anomalos)


resOutliers1 <- mvoutlier::uni.plot(datos[1:10])
resOutliers2 <- mvoutlier::uni.plot(datos[11:20])
resOutliers3 <- mvoutlier::uni.plot(datos[21:30])

anom = resOutliers1$outliers | resOutliers2$outliers | resOutliers3$outliers
datos = datos[-anom,]



## ------------------------------------------------------------------------
#datos_b = SMOTE(C~., data = datos_sin_seleccion, perc.over = 200, k=5, perc.under = 150)


## ------------------------------------------------------------------------

set.seed(3)
salidaTomek = ubTomek(datos_sin_seleccion[,-ncol(datos_sin_seleccion)], datos_sin_seleccion$C)

datosTomek = cbind(salidaTomek$X, C = salidaTomek$Y)

dim(datosTomek)

proporcion = group_by(datosTomek,C) %>% summarise(propo = round((nc = (n() * 100)/dim(datosTomek)[1]),digits = 1))

proporcion

df <- data.frame(clase=proporcion$C,
                proporcion=proporcion$propo)

p<-ggplot(df, aes(x=clase, y=proporcion, fill=clase)) +
  geom_bar(stat="identity")+theme_minimal()
p



## ------------------------------------------------------------------------
datos_b = SMOTE(C~., data = datosTomek, perc.over = 150, k=5, perc.under = 0)
datos_b = rbind(subset(datos_b, C==1), subset(datosTomek, C==0))


## ------------------------------------------------------------------------
#test = as.data.frame(scale(test, center = TRUE, scale = TRUE))
#discretizar:
#datos = discretization::disc.Topdown(datos, method=1)

modelRipp <- JRip(formula = C~.,data = datosTomek)
#prediction = predict(modelRipp, datos)

## ------------------------------------------------------------------------
#precision(predictionLabels = prediction, testLabels = datos$C)
evaluate_Weka_classifier(modelRipp, numFolds = 5, seed = 3)

## ------------------------------------------------------------------------
prediction = predict(modelRipp, test)
generarEnvio(prediction, file = "datos_tomek_con_seleccion:solo_una.csv")


