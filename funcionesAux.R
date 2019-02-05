#FUNCIONES UTILES:

# funcion para generar el archivo con el envio a kaggle. Argumentos:
# @param path ruta donde se quiere almacenar
# @param file nombre del archivo donde almacenar los datos
# @param cdataset conjunto de datos a almacenar


generarEnvio = function(y,file="envio.csv",path="./"){
  # se compone el path completo
  pathCompleto <- paste(path,file,sep="")
  
  ids <- 1:length(y)
  prediccion <- as.vector(y)
  res <- as.data.frame(cbind(ids,prediccion))
  
  write.csv(res,file = pathCompleto,row.names=FALSE,quote=FALSE)
  
  
}