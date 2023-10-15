library(xlsx)
library(ggplot2)
library(ggpubr)
library(latex2exp)
library(patchwork)
library(grid)
library(gridExtra)
library(RColorBrewer)

# Kawloon
jl_data <- read.table("Kowloon_Peninsula.txt", header = TRUE, sep = ",")
jl_data$district = as.factor(jl_data$district)

colors <- c('#eb4b3a', "#48bad0", "#1a9781", "#355783", "#ef9a80")

p_jl <- ggplot(jl_data, aes(x=value, y=district)) + 
  stat_boxplot(geom = "errorbar", size = 0.8, width = 0.2, position = position_dodge(0.6), color = colors) + 
  geom_boxplot(color = colors, size = 0.8, width = 0.6) + 
  scale_x_continuous(limits = c(0.3,0.9),
                     expand = expansion(mult=c(0,0.05)),
                     breaks = seq(0.3,0.9,0.2)) + 
  
  #  geom_jitter(aes(district, value), width = 0.01) + 
  
  scale_color_manual(values = c('#eb4b3a', "#48bad0", "#1a9781", "#355783", "#ef9a80")) + 
  ylab("Kowloon Peninsula") + xlab("") + theme_bw() + 
  theme(legend.position = "none",
        axis.text.y = element_text(angle = 0, hjust = 1.0, vjust = 0.5, color = colors),
        axis.text.x = element_blank(),
        axis.text = element_text(size = 14, face = "bold"),
        axis.title = element_text(size = 16, face = "bold")) +
  scale_fill_manual(values = c('#eb4b3a', "#48bad0", "#1a9781", "#355783", "#ef9a80")) +
  stat_summary(fun.x = mean, geom = "point", shape = 23, size = 3)

# Hong Kong Island
gd_data <- read.table("Hong_Kong_Island.txt", header = TRUE, sep = ",")
gd_data$district = as.factor(gd_data$district)

gd_colors <- c('#eb4b3a', "#48bad0", "#1a9781", "#355783")

p_gd <- ggplot(gd_data, aes(x=value, y=district)) + 
  
  stat_boxplot(geom = "errorbar", size = 0.8, width = 0.2, position = position_dodge(0.6), color = gd_colors) + 
  geom_boxplot(color = gd_colors, size = 0.8, width = 0.6) + 
  scale_x_continuous(limits = c(0.3,0.9),
                     expand = expansion(mult=c(0,0.05)),
                     breaks = seq(0.3,0.9,0.2)) + 
  
  #  geom_jitter(aes(district, value), width = 0.01) + 
  
  scale_color_manual(values = c('#eb4b3a', "#48bad0", "#1a9781", "#355783")) + 
  ylab("Hong Kong Island") + xlab("") + theme_bw() + 
  theme(legend.position = "none",
        axis.text.y = element_text(angle = 0, hjust = 1.0, vjust = 0.5, color = gd_colors),
        axis.text.x = element_blank(),
        axis.text = element_text(size = 14, face = "bold"),
        axis.title = element_text(size = 16, face = "bold")) +
  scale_fill_manual(values = c('#eb4b3a', "#48bad0", "#1a9781", "#355783")) +
  
  stat_summary(fun.x = mean, geom = "point", shape = 23, size = 3)

#grid.arrange(p_jl, p_gd, nrow = 1)

# New Territories
xj_data <- read.table("New_Territories.txt", header = TRUE, sep = ",")
xj_data$district = as.factor(xj_data$district)
xj_colors <- c('#eb4b3a', "#48bad0", "#1a9781", "#355783", "#ef9a80", "#8952A0", "#F4A2A3","#959897", "#911310")

p_xj <- ggplot(xj_data, aes(x=value, y=district)) + 
  
  stat_boxplot(geom = "errorbar", size = 0.8, width = 0.2, position = position_dodge(0.6), color = xj_colors) + 
  geom_boxplot(color = xj_colors, size = 0.8, width = 0.6) + 
  scale_x_continuous(limits = c(0.3,0.9),
                     expand = expansion(mult=c(0,0.05)),
                     breaks = seq(0.3,0.9,0.2)) + 
  
  #  geom_jitter(aes(district, value), width = 0.01) + 
  
  scale_color_manual(values = c('#eb4b3a', "#48bad0", "#1a9781", "#355783", "#ef9a80", "#8952A0", "#F4A2A3","#959897", "#911310")) + 
  ylab("New Territories") + xlab("Greening priority") + theme_bw() + 
  theme(legend.position = "none",
        axis.text.y = element_text(angle = 0, hjust = 1.0, vjust = 0.5, color = gd_colors),
        axis.text = element_text(size = 14, face = "bold"),
        axis.title = element_text(size = 16, face = "bold")) +
  scale_fill_manual(values = c('#eb4b3a', "#48bad0", "#1a9781", "#355783", "#ef9a80", "#8952A0", "#F4A2A3","#959897", "#911310")) +
  
  stat_summary(fun.x = mean, geom = "point", shape = 23, size = 3)

p_jl + p_gd + p_xj + plot_layout(nrow = 3, heights = c(5,4,9))

ggsave("fig.pdf", height = 12, width = 5)