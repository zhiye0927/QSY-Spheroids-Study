library(here)

folder_path <- here("analysis", "data", "raw_data", "core_edge_angle")

file_list <- list.files(folder_path, pattern = "\\.txt$", full.names = TRUE)

calculate_mean_angle <- function(file_path) {
  
  lines <- suppressWarnings(readLines(file_path, warn = FALSE, encoding = "UTF-8"))
  
  lines <- trimws(gsub("\u3000", " ", lines)) 
  
  data <- do.call(rbind, lapply(seq_along(lines), function(i) {
    line <- lines[i]
    match <- regmatches(line, regexec("^Pt\\d+\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)", line))
    if (length(match[[1]]) == 4) {
      return(c(as.numeric(match[[1]][2]), as.numeric(match[[1]][3]), as.numeric(match[[1]][4])))
    } else {
      stop(paste("Format mismatch, cannot skip line. Error at line:", i, "Content:", line))
    }
  }))
  
  if (is.null(data) || nrow(data) < 3) {
    warning(paste("Invalid data format or insufficient points, skipping file:", basename(file_path)))
    return(data.frame(File = basename(file_path), Mean_Angle = NA))
  }
  
  colnames(data) <- c("X", "Y", "Z")
  points <- as.matrix(data)
  
  calculate_angle <- function(A, B, C) {
    vec1 <- A - B
    vec2 <- C - B
    cos_theta <- sum(vec1 * vec2) / (sqrt(sum(vec1^2)) * sqrt(sum(vec2^2)))
    cos_theta <- max(min(cos_theta, 1), -1)
    angle_rad <- acos(cos_theta)
    angle_deg <- angle_rad * 180 / pi
    return(angle_deg)
  }
  
  angles <- c()
  for (i in seq(1, nrow(points) - 2, by = 3)) {
    A <- points[i, ]
    B <- points[i + 1, ]
    C <- points[i + 2, ]
    angle <- calculate_angle(A, B, C)
    angles <- c(angles, angle)
  }
  
  mean_angle <- if (length(angles) > 0) mean(angles) else NA
  file_name <- sub("\\.txt$", "", basename(file_path))
  return(data.frame(File = file_name, Mean_Angle = mean_angle))
}

result <- do.call(rbind, lapply(file_list, calculate_mean_angle))

print(result)

output_folder <- here("analysis", "data", "raw_data", "core_edge_angle")
output_csv <- file.path(output_folder, "mean_angles_summary.csv")
write.csv(result, file = output_csv, row.names = FALSE)

