library(ggplot2)
library('ROCR')
setwd("/Users/danielfeeney/Documents/MachineLearning")
dat <- read.csv('creditcard.csv')
#Look at the data & check for completeness
head(dat)
sum(dat$Class) #verifies 492 instances of fraud
sum(is.na(dat)) #No incomplete data

#Graphically represent the data#
attach(dat)
ggplot(dat, aes(x=V1, y=V2, col = Class)) + geom_point() 
#Zoom into region with most data
ggplot(dat, aes(x=V1, y=V2, col = Class)) + geom_point() + xlim(c(-20, 0)) + ylim(c(-25, 25))
#It seems that fraudulent cases may have a higher score on V2

#Split data into training and testing so 67% of data are in testing
num_train <- floor(dim(dat)[1] * 0.67)
train.dat <- dat[1:num_train,]
test.dat <- dat[num_train:dim(dat)[1],]

model <- glm(Class ~ V1 + V2 + V3 + V4 + V5, family=binomial(link='logit'),data=train.dat)
summary(model)
#The first four PCs are significant in predicting the test set
#Try this model on the test set

fitted.results <- predict(model,newdata= test.dat,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != test.dat$Class)
print(paste('Accuracy',1-misClasificError))

# Run model through ROC
p <- predict(model, newdata=test.dat,  type="response")
pr <- prediction(p, test.dat$Class)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
