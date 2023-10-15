library(ggplot2)
library(ggpubr)
library(latex2exp)
library(patchwork)
library(grid)


ug_data <- read.table("non_greening.txt")

p1 <- ggplot(ug_data, aes(x=V1)) +
  geom_histogram(aes(y=after_stat(count)),
                 bins = 200,
                 alpha = 0.8) +
  theme_bw() + 
  scale_x_continuous(limits = c(0,1),
                     expand = expansion(mult = c(0.02,0.02)),
                     breaks = seq(0,1,0.1)) + 
  scale_y_continuous(limits = c(0,50000),
                     expand = expansion(mult=c(0,0)),
                     breaks = seq(0,50000,10000)) + 
  theme(axis.text = element_text(size = 12),
        axis.text.x = element_text(angle = 0),
        axis.text.y = element_text(angle = 45),
        strip.text = element_text(size = 12),
        axis.title = element_text(size = 14)) +
  xlab("Greenspace coverage rate (%)") + ylab("Roof points") + 
  geom_vline(xintercept = 0.35, lty = "dashed", size = 1)


g_data <- read.table("greening.txt")
p2 <- ggplot(g_data, aes(x=V1)) +
  geom_histogram(aes(y=after_stat(count)),
                 bins = 250,
                 alpha = 0.8) +
  theme_bw() + 
  scale_x_continuous(limits = c(0,1),
                     expand = expansion(mult = c(0.02,0.02)),
                     breaks = seq(0,1,0.1)) + 
  scale_y_continuous(limits = c(0,50000),
                     expand = expansion(mult=c(0,0)),
                     breaks = seq(0,50000,10000)) + 
  theme(axis.text = element_text(size = 12),
        axis.text.x = element_text(angle = 0),
        axis.text.y = element_blank(),
        strip.text = element_text(size = 12),
        axis.title = element_text(size = 14)) +
  xlab("Greenspace coverage rate (%)") + ylab("Roof points") + 
  geom_vline(xintercept = 0.58, lty = "dashed", size = 1)

remove_y <- theme(axis.text.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  axis.title.y = element_blank())
p <- list(p1,
          p2 + remove_y)
wrap_plots(p,nrow = 1) + plot_layout(guides = "collect")