library(mxnet)
source("dcca.R")

set.seed(123456)
x <- runif(1000,-1,1)
y <- sin(5*x)

plot(x,y,col=col,pch=19,cex=0.5,xlab="deep(X)",ylab="y")

library(gplots)
library(grDevices)
col <- colorpanel(length(y),"blue","yellow","red")
col <- adjustcolor(col,0.8)[rank(x)]


dcca <- deepCCA(x,y,col=col)


set.seed(1)
x1 <- runif(1000,-1,1)
set.seed(2)
x2 <- runif(1000,-1,1)
set.seed(3)
x3 <- runif(1000,-1,1)
x <- cbind(x1,x2,x3)
y <- x1*x2*x3

col <- colorpanel(length(y),"blue","yellow","red")
col <- adjustcolor(col,0.8)[rank(y)]

dcca <- deepCCA(x,y,col=col)




