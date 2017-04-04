sp <- ggplot(as.data.frame(ROM.dat), aes(x= ROM.dat[,3], y =ROM.dat[,2], col= factor(ROM.dat[,1]), group = ROM.dat[,1])) + geom_point() + geom_line() +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.background = element_blank()) 
sp + theme(legend.position='none') + ylab("Change in range of motion") + xlab("Treatment") +
  scale_x_continuous(breaks = c(1,2,3), labels=c("Control", "TENS", "Balls"))
