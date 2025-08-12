library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)

# 读取 Excel 文件
file_path <- "analysis/data/derived_data/umap_l2_l3.XLSX"
df <- read_excel(here(file_path))

# 查看数据结构
print(head(df))

# 计算相关系数矩阵
correlations <- df %>%
  select(UMPA1, UMAP2, degree1, degree2) %>%
  cor()

print("相关系数矩阵：")
print(correlations)

# 将数据长格式化方便绘图
df_long <- df %>%
  pivot_longer(cols = c(degree1, degree2), names_to = "degree", values_to = "degree_value")

# 绘制 UMAP1 和 degree的散点图
p1 <- ggplot(df_long, aes(x = UMPA1, y = degree_value, color = degree)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "UMAP1 vs degree1 (l=2) and degree2 (l=3)",
       x = "UMAP1",
       y = "Degree Power") +
  theme_minimal()

# 绘制 UMAP2 和 degree的散点图
p2 <- ggplot(df_long, aes(x = UMAP2, y = degree_value, color = degree)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "UMAP2 vs degree1 (l=2) and degree2 (l=3)",
       x = "UMAP2",
       y = "Degree Power") +
  theme_minimal()

print(p1)
print(p2)
