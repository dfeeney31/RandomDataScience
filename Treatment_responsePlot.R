sp <- ggplot(as.data.frame(ROM.dat), aes(x= ROM.dat[,3], y =ROM.dat[,2], col= factor(ROM.dat[,1]), group = ROM.dat[,1])) + geom_point() + geom_line() +
  ggtitle("The magical effects of balls") +
  theme_bw() +
  theme(plot.title = element_text(face = 'bold'),
        axis.line = element_line(colour = "black", size =2),
        axis.text = element_text(size = 12),
        panel.background = element_blank()) 
sp + theme(legend.position='none') + ylab("Change in torque") + xlab("Treatment") +
  scale_x_continuous(breaks = c(1,2,3), labels=c("Control", "TENS", "Balls"))
