library(dplyr)
library(Hmisc)
library(mice)
library(Amelia)
library(PerformanceAnalytics)
library(caret)

train = read.csv("train.csv", na.string=c(" ", "NA", "?"))
test = read.csv("test.csv", na.string=c(" ", "NA", "?"))

train = as_tibble(train)
test = as_tibble(test)

describe(train)


ggplot(data=train) + geom_bar(aes(x=C, y = ..prop.., group=1))


patrin = md.pattern(x = train[1:200, 10])


missmap(train[1:300, 1:20], col = c("red", "steelblue"))


resumen <- dplyr::group_by(train,C) %>%dplyr::summarise(nc=n())
resumen


patron <- mice::md.pattern(x=train[1:200,1:20],plot=TRUE)
completos <- mice::ncc(train)
incompletos <- mice::nic(test)


#Imputación:

imputacion = mice(train, m=1, method = "cart", printFlag = TRUE)
incompletas = nic(imputados)
incompletas

imputados = amelia(train, m=1, parallel= "multicore", noms = "C")
nic(imputados$imputations$imp1)

corrMatrix = cor(na.omit(train))

corrplot::corrplot(corrMatrix, type = "upper", order="FPC", tl.col = "black", tl.srt = 45)
chart.Correlation(na.omit(train[,1:5]), histogram = TRUE)

#filtrado de variables:
altamenteCorreladas = caret::findCorrelation(corrMatrix, cutoff=0.8)

filtrado = train [, -altamenteCorreladas]
filtrado = as_tibble(filtrado)


#reglas jripper y oneR:
library(OneR)

modelo1R = OneR(C~., data = train)
modelo1R = OneR(C~., data = filtrado)

#discretizamos.
filtradoDiscretizado = optbin(as.data.frame(filtrado))
modelo2R = OneR(C~., data = filtradoDiscretizado)


prediccion = predict(modelo2R, as.data.frame(test))
prediccion = as.vector(prediccion)

ids = 1:nrow(test)
res = as.data.frame(cbind(ids, prediccion))
colnames(res) = c("Id", "Prediccion")

write.csv(res, file ="./envio.csv", row.names = FALSE, quote = FALSE)

#jripper:
library(RWeka)

train$C =as.factor(train$C) 
modeloJR = JRip(C~., data = train)

evaluate_Weka_classifier(modeloJR, numFolds = 10)

# arboles de clasificación
modeloJ48 = J48(C~., data = train)

evaluate_Weka_classifier(modeloJ48, numFolds = 10)
library(gdm)
library(mlbench)
data(Sonar)

enTrain = createDataPartition(Sonar$Class, p = .75, list = FALSE)
train = Sonar[enTrain,]
test = Sonar[-enTrain,]

fitcontrol = trainControl(method = "repeatedcv", number = 5, repeats = 3)

modelo = train(Class~., data = train, method = "gbm", trControl=fitcontrol, verbose=FALSE)
grid = expand.grid(interaction.depth=c(1,5,9), n.trees = c(1:10)*50, shrinkage =0.1, n.minobsinnode=20)

modelo2 = train(Class~., data = train, method= "gbm", trControl=fitcontrol, verbose=FALSE, tuneGrid = grid)
