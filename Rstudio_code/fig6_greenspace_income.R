library(xlsx)
library(ggpubr)
library(ggplot2)
library(ggpmisc)
library(patchwork)
library(tidyverse)
library(reshape2)
library(RColorBrewer)

gi_data = read.xlsx("income_greenspace.xlsx", sheetIndex = 1)

p1 <- ggplot(gi_data) +
  geom_line(aes(no, income), size=1.5,color="#0072b2") + 
  scale_x_continuous(breaks = seq(0,540,36)) + theme_bw() + 
  scale_y_continuous(breaks = seq(0,120000,30000)) + 
  theme(axis.text = element_text(size = 12),
#        axis.text.x = element_text(angle = 45),
        axis.text.x = element_blank(),
        axis.text.y = element_text(angle = 90),
        strip.text = element_text(size = 12),
        axis.title = element_text(size = 14)) +
  xlab("") + ylab("Monthly income (HK$)")
p2 <- ggplot(gi_data) +
  geom_line(aes(no, greenspace), size=1, color="#d55e00") + 
  scale_x_continuous(breaks = seq(0,540,36)) + theme_bw() + 
  theme(axis.text = element_text(size = 12),
        axis.text.x = element_text(angle = 0),
        axis.text.y = element_text(angle = 90),
        strip.text = element_text(size = 12),
        axis.title = element_text(size = 14)) +
  xlab("No.") + ylab("Greenspace rate (%)")

#library(cowplot)
#plot_grid(p1,p2,ncol = 1)

data = read.xlsx("income_greenspace_standardization.xlsx", sheetIndex = 1)

p3 <- ggplot(data,aes(x=green, y=income)) +
  geom_smooth(method = "lm", color = "Red", fill = "lightgray", alpha = 0.8) + 
  stat_cor(data=data,method = "pearson") + 
  xlab("Greenspace coverage rate (%)") + ylab("Monthly median income (%)") + 
  theme_bw() + theme(axis.text = element_text(size = 12),
                   axis.text.x = element_text(angle = 0),
                   strip.text = element_text(size = 12),
                   axis.title = element_text(size = 14)) +
  geom_point(colour = "black", size=2, alpha = 0.6, shape = 19)

library(grid)
grid.newpage()
pushViewport(viewport(layout = grid.layout(2,7)))
vplayout <- function(x,y){viewport(layout.pos.row = x,layout.pos.col = y)}
print(p1, vp=vplayout(1,1:4))
print(p2, vp=vplayout(2,1:4))
print(p3, vp=vplayout(1:2,5:7))
pdf("fig6.pdf", height = 4, width = 10)
print(p1)
print(p2)
print(p3)
dev.off()