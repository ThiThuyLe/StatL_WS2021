
# ------------------------------------------------------------------------------
# Published in:     StatL_class_WS2021
# ------------------------------------------------------------------------------
# Project title:    P2P Lending Club
# ------------------------------------------------------------------------------
# Description:      The code does the following tasks:
#                   - Read in data set
#                   - Missing values identification
#                   - Install and load required packages
#                   - Split data set into training and test randomly
#                   - Perform step wise model selection by AIC
#                   - Fit training data using linear regression, logistic
#                     regression, and support vector machine (SVM) with
#                     different kernel functions
#                   - Select best parameters for SVM through 10-fold cross
#                     validation
#                   - Obtain predictions of test set
#                   - Plot linear model, residual, and logistic regression
#                     model
#                   - Create confusion matrices
# ------------------------------------------------------------------------------
# Output:           - Plot of linear regression model
#                   - Residual plot
#                   - Plot of logistic regression model
#                   - Confusion matrices (logistic regression, SVM)
# ------------------------------------------------------------------------------
# Author :          Hans Alcahya Tjong (570795)
#                   Ananda Eraz Irfananto (547048)
# ------------------------------------------------------------------------------
# Dataset:          "lendingclub.csv"
# ------------------------------------------------------------------------------

# Don't forget to change the file directory
#setwd("/Users/hanstjong/OneDrive/HTW/B-WM/6. Sem/Seminar/Projekt")
#setwd("~/HTW/WMATHE/SEMESTER4/WISEARBEIT")

# Read in data set
data_00 <- read.csv("lendingclub.csv"); head(data_00)
data_00$credit.policy = factor(data_00$credit.policy)

# Size of observation and feature
dim(data_00)

# Missing values identification and handling
sum(is.na(data_00))
# if sum == 0, then no missing value.
# if sum != 0, remove observation

# Install and load packages
libraries = c("caret","LogicReg","e1071","MASS", "tibble", "ggplot2")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# Split randomly into training and test data with ratio 70:30
set.seed(1234)
indizes = createDataPartition(y = data_00$credit.policy, p = 0.70, list = F)
data_train <- data_00[indizes,]
data_test <- data_00[-indizes,]
# ------------------------------------------------------------------------------
# Linear Regression
lm_fit = lm(fico~., data = data_train)
summary(lm_fit)

# Step AIC
AIC(lm_fit)
steplm = stepAIC(lm_fit, direction = "both")
steplm$anova

# Modified linear regression
lm_fit1 = lm(fico~credit.policy + purpose + int.rate + installment +
               log.annual.inc + dti + days.with.cr.line + revol.bal +
               revol.util + delinq.2yrs + pub.rec + not.fully.paid,
               data = data_train)
summary(lm_fit1)

# Predict on test set
lm_prob = predict(lm_fit1, newdata = data_test, type = "response")
lm_prob1 = enframe(lm_prob, name = "applicants", value = "fico")

# Plot linear model
plot(lm_prob1$fico, data_test$fico, xlab = "FICO (predicted)",
     ylab = "FICO (actual)")
abline(0,1, col = "red")

# Residual Plot
residual = data_test$fico - lm_prob1$fico
plot(lm_prob1$fico, residual, ylab = "Residual", xlab = "FICO (test)")
abline(0,0, col = "red")
# ------------------------------------------------------------------------------
# Logistic Regression
glm_fit = glm(credit.policy ~ ., data = data_train, family = binomial)
summary(glm_fit)

# AIC
step = stepAIC(glm_fit, direction  = "both")
step$anova

# Modified logistic regression (ignore the warning)
fit.log1 <- glm(credit.policy ~ installment + log.annual.inc + fico +
                days.with.cr.line + revol.bal +  delinq.2yrs + not.fully.paid +
                revol.util + inq.last.6mths, data = data_train, family=binomial)
summary(fit.log1)

# Predict on test set
glm.prob1 = predict(fit.log1, data_test, type = "response");head(glm.prob1)
glm_prob1frame = enframe(glm.prob1, name = "applicants", value = "probability")

# Probability > 0.5 --> 1, otherwise 0
glm.pred1 = factor(ifelse(glm.prob1>0.5,1,0));head(glm.pred1)

# Confusion matrix
confusionMatrix(glm.pred1, data_test$credit.policy)

# Plot
for(i in 1:nrow(data_test)){
  f <- 0
  f[i] <- data_test$fico[i]
  glm_prob1frame$fico[i] <- f[i]
}
head(glm_prob1frame)


pl <- ggplot(glm_prob1frame, aes(x=fico, y =probability))
pl + 
  geom_point( ) +
  ggtitle("Probability Credit Policy by FICO") + 
  ggplot2::ylab("Probability Credit Policy")+
  theme(
    panel.background = element_rect(fill='transparent'),
    legend.background = element_rect(fill='transparent'),
    legend.box.background = element_rect(fill='transparent')
  )

# Modified logistic regression (Consider the warning)
fit.log2 <- glm(credit.policy ~ installment + log.annual.inc + fico +
                days.with.cr.line + revol.bal +  delinq.2yrs + not.fully.paid,
                data = data_train, family=binomial)
summary(fit.log2)

# Predict on test set
glm.prob2 = predict(fit.log2, data_test, type = "response");head(glm.prob2)
glm.pred2 = factor(ifelse(glm.prob2>0.5,1,0));head(glm.pred2)

# Confusion matrix
confusionMatrix(glm.pred2, data_test$credit.policy)

# take warning into consideration --> less accuracy
# ------------------------------------------------------------------------------
# Support Vector Machines
# Kernel function = radial
svmfit = svm(credit.policy ~ installment + log.annual.inc + fico +
            days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
            delinq.2yrs + not.fully.paid, data = data_train, kernel="radial",
            gamma=1,cost=1)
summary(svmfit)

# Best parameter
tune.out = tune.svm(credit.policy ~ installment + log.annual.inc + fico +
                  days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
                  delinq.2yrs + not.fully.paid, data = data_train,
                  kernel="radial", cost=c(0.1,1,10,100,1000),
                  gamma=c(0.5,1,2,3,4))
summary(tune.out)

# SVM with best parameters
svmfit_b = svm(credit.policy ~ installment + log.annual.inc + fico +
              days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
              delinq.2yrs + not.fully.paid, data = data_train, kernel="radial",
              gamma=0.5,cost=10)
summary(svmfit_b)

# Predict on test set
svm.pred = predict(svmfit_b, data_test);table(svm.pred)
confusionMatrix(svm.pred, data_test$credit.policy)

# Kernel function = linear
svmfit_l = svm(credit.policy ~ installment + log.annual.inc + fico +
              days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
              delinq.2yrs + not.fully.paid, data = data_train, kernel="linear",
              cost=1)
summary(svmfit_l)

# Best parameter
tune.out_l = tune.svm(credit.policy ~ installment + log.annual.inc + fico +
                  days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
                  delinq.2yrs + not.fully.paid, data = data_train,
                  kernel="linear", cost=c(0.001,0.01,0.1,1,10))

summary(tune.out_l)

# SVM with best parameters
svmfit_l_b = svm(credit.policy ~ installment + log.annual.inc + fico +
                days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
                delinq.2yrs + not.fully.paid, data = data_train,
                kernel="linear", cost=0.01)
summary(svmfit_l_b)

# Predict on test set
svm.pred_l_b = predict(svmfit_l_b, data_test);table(svm.pred_l_b)
confusionMatrix(svm.pred_l_b, data_test$credit.policy)

# Kernel function = polynomial, d = 2
svmfit_p2 = svm(credit.policy ~ installment + log.annual.inc + fico +
              days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
              delinq.2yrs + not.fully.paid, data = data_train, 
              kernel="polynomial", degree = 2)
summary(svmfit_p2)

# Predict on test set
svm.pred_p2 = predict(svmfit_p2, data_test);table(svm.pred_p2)
confusionMatrix(svm.pred_p2, data_test$credit.policy)

# Best parameter
tune.out_p2 = tune.svm(credit.policy ~ installment + log.annual.inc + fico +
                  days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
                  delinq.2yrs + not.fully.paid, data = data_train,
                  kernel="polynomial", cost=c(0.5,1,2), gamma=c(.1,1), degree=2)
summary(tune.out_p2)

# SVM with best parameters
svmfit_p2_b = svm(credit.policy ~ installment + log.annual.inc + fico +
                days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
                delinq.2yrs + not.fully.paid, data = data_train,
                kernel="polynomial", cost=0.5, gamma=1, degree=2)
summary(svmfit_p2_b)

# Predict on test set
svm.pred_p2_b = predict(svmfit_p2_b, data_test);table(svm.pred_p2_b)
confusionMatrix(svm.pred_p2_b, data_test$credit.policy)


# Kernel function = polynomial, d = 3
svmfit_p3 = svm(credit.policy ~ installment + log.annual.inc + fico +
              days.with.cr.line + revol.bal + revol.util + inq.last.6mths +
              delinq.2yrs + not.fully.paid, data = data_train,
              kernel="polynomial", degree = 3, cost=1, gamma=1)
summary(svmfit_p3)

# Predict
svm.pred_p3 = predict(svmfit_p3, data_test);table(svm.pred_p3)
confusionMatrix(svm.pred_p3, data_test$credit.policy)
# ------------------------------------------------------------------------------