#FUNCIONES UTILES:

# funcion para generar el archivo con el envio a kaggle. Argumentos:
# @param path ruta donde se quiere almacenar
# @param file nombre del archivo donde almacenar los datos
# @param cdataset conjunto de datos a almacenar


generarEnvio = function(y,file="envio.csv",path="./"){
  # se compone el path completo
  pathCompleto <- paste(path,file,sep="")
  
  Id <- 1:length(y)
  Prediction <- as.vector(y)
  res <- as.data.frame(cbind(Id,Prediction))
  
  write.csv(res,file = pathCompleto,row.names=FALSE,quote=FALSE)
  
  
}


# funcion para generar el archivo con el envio a kaggle. Argumentos:
# @param predictionLabels etiquetas predichas por el modelo de prediccion
# @param testLabels etiquetas de test
# @output precision

precision = function(predictionLabels, testLabels){
  
  tabla = table(predictionLabels,testLabels)
  (tabla[1,1] + tabla[2,2] )/ sum(tabla)
}