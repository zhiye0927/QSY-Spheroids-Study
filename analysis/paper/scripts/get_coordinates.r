

get_coordinates <- function (filename, stop_pattern) {
  
  temp_file <- readLines(filename)
  
  xyz1 <- as.data.frame(matrix(ncol=3))
  colnames(xyz1) <- c("X1", "Y1", "Z1")
  
  xyz2 <- as.data.frame(matrix(ncol=3))
  colnames(xyz2) <- c("X2", "Y2", "Z2")
  
  
  for (i in 1:length(temp_file)) {
    
    temp_line <- temp_file[i]
    
    if (substr(temp_line, start=5, stop=10) == "CV[ 0]") {
      
      coordinates <- gsub("\\(([^()]*)\\)|.", "\\1", temp_line, perl=T)
      
      x <- as.numeric(gsub("^(.*?),.*", "\\1", coordinates))
      y <- as.numeric(sub("^([^,]+),\\s*([^,]+),.*", "\\2", coordinates))
      z <- as.numeric(sub('.*,\\s*', '', coordinates))
      
      xyz <- as.data.frame(t(c(x, y, z)))
      colnames(xyz) <- c("X1", "Y1", "Z1")
      #print(xyz)
      xyz1 <- rbind(xyz1, xyz)
      
    }
    
    if (substr(temp_line, start=5, stop=10) == "CV[ 1]") {
      
      coordinates <- gsub("\\(([^()]*)\\)|.", "\\1", temp_line, perl=T)
      
      x <- as.numeric(gsub("^(.*?),.*", "\\1", coordinates))
      y <- as.numeric(sub("^([^,]+),\\s*([^,]+),.*", "\\2", coordinates))
      z <- as.numeric(sub('.*,\\s*', '', coordinates))
      
      xyz <- as.data.frame(t(c(x, y, z)))
      colnames(xyz) <- c("X2", "Y2", "Z2")
      #print(xyz)
      xyz2 <- rbind(xyz2, xyz)
      
    }
    
    
  }
  
  stop.pattern <- paste0(".*",stop_pattern)
  
  xyz <- round(cbind(xyz1[-1,], xyz2[-1,]), 5)
  temp_name <- sub(stop.pattern, '', filename)
  xyz$ID <- substr(temp_name, start = 1, stop = 2)
  xyz$type <- str_sub(temp_name, start=paste0("-",(nchar(temp_name)-3)), end=nchar(temp_name)-4)
    substr(sub("\\_.*", "", filename), 4, nchar(sub("\\_.*", "", filename)))
  xyz$length <- sqrt((xyz$X2 - xyz$X1)^2 + (xyz$Y2 - xyz$Y1)^2 + (xyz$Z2 - xyz$Z1)^2)
  xyz$length_scaled <- xyz$length / sum(xyz$length)
  
  return(xyz)
  #print(xyz$ID)
  
  
}
