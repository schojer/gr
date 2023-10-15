library(xlsx)
library(ggpubr)
library(ggplot2)
library(ggpmisc)
library(patchwork)
library(tidyverse)
library(reshape2)
library(RColorBrewer)

#read data
un_data = read.xlsx("D:/Develop/R/Data/wrf/ungreen.xlsx", sheetIndex = 1)
gr_data = read.xlsx("D:/Develop/R/Data/wrf/green.xlsx", sheetIndex = 1)
dif_data = read.xlsx("D:/Develop/R/Data/wrf/diffs.xlsx", sheetIndex = 1)

#create a dataset
lintypes <- c(1,4,1,4,1,4,1,4)

p1 <- ggplot() + 
  
  # ungreening
  geom_line(data = un_data, aes(x=time, y=spring, color = "BR_Spring"), lwd = 1.1, lty = 1) +
  geom_line(data = un_data, aes(x=time, y=summer, color = "BR_Summer"), lwd = 1.1, lty = 1) +
  geom_line(data = un_data, aes(x=time, y=autumn, color = "BR_Autumn"), lwd = 1.1, lty = 1) +
  geom_line(data = un_data, aes(x=time, y=winter, color = "BR_Winter"), lwd = 1.1, lty = 1) +
  
  #Greening
  geom_line(data = gr_data, aes(x=time, y=spring, color = "GR_Spring"), lwd = 1.1, lty = 4) +
  geom_line(data = gr_data, aes(x=time, y=summer, color = "GR_Summer"), lwd = 1.1, lty = 4) +
  geom_line(data = gr_data, aes(x=time, y=autumn, color = "GR_Autumn"), lwd = 1.1, lty = 4) + 
  geom_line(data = gr_data, aes(x=time, y=winter, color = "GR_Winter"), lwd = 1.1, lty = 4) + 
  
  scale_color_manual("",values = c("BR_Spring" = "#349839",
                                   "BR_Summer" = "#EA5D2D",
                                   "BR_Autumn" = "#EABB77",
                                   "BR_Winter" = "#2072A8",
                                   "GR_Spring" = "#349839",
                                   "GR_Summer" = "#EA5D2D",
                                   "GR_Autumn" = "#EABB77",
                                   "GR_Winter" = "#2072A8"),
                     limits = c("BR_Spring", "GR_Spring",
                                "BR_Summer", "GR_Summer",
                                "BR_Autumn", "GR_Autumn",
                                "BR_Winter", "GR_Winter")) + 
  
  xlab("Time") + ylab("Air temperature (°C)") +
  theme_bw() + 
  scale_x_continuous(limits = c(0,24),
                     expand = expansion(mult = c(0,0)),
                     breaks = seq(0,24,4)) + 
  scale_y_continuous(limits = c(11,34),
                     expand = expansion(mult = c(0,0.01)),
                     breaks = seq(11,34,4.6)) + 
  
  theme(axis.text = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(angle = 0),
        strip.text = element_text(size = 12, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 10, face = "bold"),
        legend.position = c(0.5,0.075),
        legend.background = element_rect(fill = rgb(1,1,1,alpha = 0.001))) +
  guides(colour= guide_legend(override.aes = list(lty = lintypes), nrow=2))

p2 <- ggplot() + 
  
  geom_line(data = dif_data, aes(x=time, y=spring, color = "Spring"), lwd = 1.1, lty = 1) +
  geom_line(data = dif_data, aes(x=time, y=summer, color = "Summer"), lwd = 1.1, lty = 1) +
  geom_line(data = dif_data, aes(x=time, y=autumn, color = "Autumn"), lwd = 1.1, lty = 1) +
  geom_line(data = dif_data, aes(x=time, y=winter, color = "Winter"), lwd = 1.1, lty = 1) +
  scale_color_manual("",values = c("Spring" = "#349839",
                                   "Summer" = "#EA5D2D",
                                   "Autumn" = "#EABB77",
                                   "Winter" = "#2072A8"),
                     limits = c("Spring", "Summer",
                                "Autumn", "Winter")) + 
  xlab("Time") + ylab("Air temperature reduction (°C)") +
  theme_bw() + 
  scale_x_continuous(limits = c(0,24),
                     expand = expansion(mult = c(0,0)),
                     breaks = seq(0,24,4)) + 
  scale_y_continuous(limits = c(-0.45,0.2),
                     expand = expansion(mult = c(0,0.01)),
                     breaks = seq(-0.45,0.2,0.13)) + 
  theme(axis.text = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(angle = 0),
        strip.text = element_text(size = 12, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 10, face = "bold"),
        legend.position = c(0.5,0.05),
        legend.background = element_rect(fill = rgb(1,1,1,alpha = 0.001))) +
  guides(col = guide_legend(nrow = 1, byrow = T)) +
  geom_hline(yintercept = 0, lty = 2, size = 1, color = "gray")

library(grid)
p <- list(p1, p2)
wrap_plots(p,nrow = 2)# + plot_layout(guides = "collect")
ggsave("fig5bc.pdf",width = 5.8, height = 11.5)
