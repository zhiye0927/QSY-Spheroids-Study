vector_stats <- function(xdata, field="ID") {
  
  output <- as.data.frame(matrix(ncol=6))
  colnames(output) <- c("ID", "total_distance", "resultant_magnitude", "vector_ratio", "mean_angle", "mean_distance")
  
   ID <- xdata$ID[1]
  
   xdata$PX <-xdata$X2-xdata$X1
   xdata$PY <-xdata$Y2-xdata$Y1
   xdata$PZ <-xdata$Z2-xdata$Z1
    
    resultant_vector_X <- sum(xdata$PX)
    resultant_vector_Y <- sum(xdata$PY)
    resultant_vector_Z <- sum(xdata$PZ)
    
    resultant_magnitude <- sqrt(resultant_vector_X^2 + resultant_vector_Y^2 + resultant_vector_Z^2)
    
    total_distance <- sum(xdata$length)
    
    vector_ratio <- resultant_magnitude / total_distance
    
    mean_angle <- NISTradianTOdeg(acos(vector_ratio))
    
   xdata$mid_point_z <- (xdata$Z1 +xdata$Z2)/2
    
    mean_distance <- mean(abs(xdata$mid_point_z), na.rm=TRUE)
    
    output <- rbind(output, c(ID, total_distance, resultant_magnitude, vector_ratio, mean_angle, mean_distance))
    
  output <- output[-1,]
  rownames(output) <- 1:nrow(output)
  output$total_distance <- as.numeric(output$total_distance)
  output$resultant_magnitude <- as.numeric(output$resultant_magnitude)
  output$vector_ratio <- as.numeric(output$vector_ratio)
  output$mean_angle <- as.numeric(output$mean_angle)
  output$mean_distance <- as.numeric(output$mean_distance)
  output
  
} 