
library(here)
data_dir <- here("analysis", "data", "derived_data", "power_spectrum")
csv_files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)

# 读取所有 CSV 文件，并加上文件名
all_data <- map_dfr(csv_files, function(file) {
  df <- read.csv(file, header = FALSE)
  colnames(df) <- c("degree", "total_power", "max_amplitude", "total_amplitude")
  df$file <- basename(file)
  return(df)
})


# 添加原料分组
all_data <- all_data %>%
  mutate(
    material = case_when(
      str_detect(file, regex("0562|0700", ignore_case = TRUE)) ~ "Other",
      str_detect(file, regex("0579|0680|1111", ignore_case = TRUE)) ~ "Chert",
      str_detect(file, regex("435|186|140|2600|2611|1777|1307", ignore_case = TRUE)) ~ "Dolomite",
      TRUE ~ "Lava"
    )
  )

# 计算每组的 degree 平均值和标准差
summary_df <- all_data %>%
  group_by(material, degree) %>%
  summarise(
    mean_power = mean(total_power, na.rm = TRUE),
    sd_power = sd(total_power, na.rm = TRUE),
    .groups = "drop"
  )

# 绘图
p_power_spectrum <- ggplot(summary_df, aes(x = degree, y = mean_power, color = material, fill = material)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(
    ymin = pmax(mean_power - sd_power, 1e-10),
    ymax = mean_power + sd_power
  ), alpha = 0.2, linetype = 0) +
  scale_y_log10(
    breaks = c(1e-8, 1e-6, 1e-4, 1e-2, 1),
    labels = parse(text = c("10^-8", "10^-6", "10^-4", "10^-2", "10^0"))
  ) +
  scale_x_continuous(
    limits = c(0, 20),
    breaks = seq(0, 20, by = 4),
    expand = c(0, 0)
  ) +
  coord_cartesian(clip = "on") +
  scale_color_manual(values = c(
    "Lava" = "orange",
    "Dolomite" = "#b2abd2",
    "Chert" = "#66c2a5",
    "Other" = "#fc8d62"
  )) +
  scale_fill_manual(values = c(
    "Lava" = "orange",
    "Dolomite" = "#b2abd2",
    "Chert" = "#66c2a5",
    "Other" = "#fc8d62"
  )) +
  labs(
    x = "Degree (l)",
    y = "Mean Total Power (log scale)",
    color = "Material",
    fill = "Material"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10),
    legend.position = c(0.95, 0.4),
    legend.justification = c("right", "top"),
    legend.background = element_rect(fill = alpha("white", 0.8), color = NA),
    legend.box.background = element_rect(color = "gray80"),
    legend.box.margin = margin(5, 5, 5, 5)
  )

p_power_spectrum

p_power_spectrum

