Vdata <- read.csv("../data/videogames.csv")
#Vdata <- Vdata[as.numeric(as.character(Vdata$Year_of_Release)) > 2009, ]
Vdata <- Vdata[(!is.na(Vdata$Critic_Score)),]
rangef <- function(x, divider=5) {
  x %/% divider * divider + divider/2
}

Anova_test <- function(classifier, data = Vdata) {
  summary(aov(data$Global_Sales ~ data[[classifier]]))
}

testclass <- function(classifier, data=Vdata){
  table <- tapply(Vdata$Global_Sales, Vdata[[classifier]], function(x){x})
  table[[1]] <- NULL
  par(mfrow = c(1, 2))
  boxplot(table, las = 3, cex=0.2, pch=20)
  boxplot(table, ylim=c(0,5), las = 3, cex=0.2,pch=20)
  Anova_test(classifier)
}

get_freq <- function(x, breaks) {
  len <- length(breaks)
  res <- numeric(len)
  for (i in seq_len(len)) {
    res[i] <- sum(breaks[i] < x & x <= breaks[i+1])
  }
  res
}

get_density <- function(x, breaks, by) {
  res <- get_freq(x, breaks) / (length(x) * by)
  res
}

get_density_idx <- function(gs) {
  ceiling(20 * gs)
}

rangef <- function(x, divider=5) {
  x %/% divider * divider + divider/2
}

testclass(4) #genre
testclass(2) #platform
testclass(5) #publisher

# Critic - Sale
plot((Vdata$Critic_Score),Vdata$Global_Sales, cex=0.1)
x <- seq(0,100,len=1000)
y <- 1/1500*x^2
lines(x,y,lty=3,col="red")
legend("topleft", "1/1500 * x^2", lty=2, col="red", inset=0.05)
idx <- Vdata$Global_Sales <= (Vdata$Critic_Score)^2/1500
plot((Vdata$Critic_Score)[idx],Vdata$Global_Sales[idx], cex=0.1)
plot(factor((Vdata$Critic_Score)[idx]), ylim=c(0,5),Vdata$Global_Sales[idx], cex=0.1)
f <- 1/20000000*x^4+1/1500000000*x^5+0.2
t <- 1/50000000*x^4
m <- 1/80000000*x^4+1/2000000000*x^5+0.1
lines(x,f,lty=2,col="red")
lines(x,t,lty=2,col="blue")
lines(x,m,lty=2,col="green")

cor.test(as.numeric(as.character(Vdata$User_Score)), Vdata$Critic_Score)


# 5 range
Vdata <- Vdata[idx, ]
Vdata <- Vdata[Vdata$Global_Sales <= 4, ]
ran <- rangef((Vdata$Critic_Score)[idx])
plot(factor(ran), ylim=c(0,5),Vdata$Global_Sales[idx], cex=0.1)
hist(Vdata$Global_Sales[ran==82.5], breaks=100, col="red", add=TRUE)
hist(Vdata$Global_Sales[ran==77.5], breaks=100, col = 'blue')
hist(Vdata$Global_Sales[ran==72.5], breaks=100, add=TRUE,col = 'green')

legend("topleft", "1/1500 * x^2", lty=2, col="red", inset=0.05)



# Uscore - Sale
plot(as.numeric(as.character(Vdata$User_Score)), Vdata$Global_Sales, cex=0.1)
x <- seq(0,100,len=1000)
y <- 1/1500*x^2
lines(x,y,lty=3,col="red")
legend("topleft", "1/1500 * x^2", lty=2, col="red", inset=0.05)
plot(as.numeric(as.character(Vdata$User_Score)), Vdata$Global_Sales, cex=0.1)
z <- (exp(1/50*x)-1)
lines(x,z,lty=3,col="blue")
legend("topleft", "exp(1/50*x)-1", lty=2, col="blue", inset=0.05)

