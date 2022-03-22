##### SETTINGS #####
#Seminar s0569723
#Thi Thanh Tu Phan

install.packages('glmnet')
library(glmnet)
library(datasets)
library(ggplot2)
rm(list = ls())

#Datensatz einlesen:

dataset = read.table("~/Documents/PM10 - Kopie.txt")
head(dataset)

##### Datenanlyse #####

#Namen für einzelne Spalte eingeben
colnames (dataset) = c("PM10","AnzahlAutos","Temperatur","Windgeschwindigkeit","Temperaturdifferenz", "Windrichtung", "Tagesstunde","Tagesnummer")
PM10               = (dataset[1])
AnzahlAutos        = (dataset[2])  
Temperatur         = (dataset[3])
Windgeschwindigkeit= (dataset[4])
Temperaturdifferenz= (dataset[5])
Windrichtung       = (dataset[6])
Tagesstunde        = (dataset[7])
Tagesnummer        = (dataset[8])


# Missings
any(is.na(dataset))

# visualization


####### Multiple Lineare Regression #####

# Modell für multiple lineare Regression
mlm      = lm(PM10 ~ AnzahlAutos+ Temperatur + Windgeschwindigkeit + Temperaturdifferenz + Windrichtung + Tagesstunde + Tagesnummer,  data = dataset)
summary(mlm) 



##### Lasso Regression #####
# Spliting  dataset into two parts based on outcome: 70% and 30%
set.seed(89)
train.   = sample(1:nrow(dataset), nrow(dataset)*0.70)
test.    = (-train)
trainset = dataset[train,]
testset  = dataset[test,]
x        = model.matrix(PM10~.,dataset)[,-1]
y        = dataset$PM10
y.test   =  y[test]

# fit der Lasso Regression
grid      = 10^seq(10,-2,length=1000)
lasso.mod = glmnet(x[train,],y[train],alpha=1,lambda=grid)
plot(lasso.mod)

set.seed(1)
cv.out    = cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam   = cv.out$lambda.min
bestlam

#verwende best lambda für Variablenselektion

out        = glmnet(x,y,alpha=1,lambda=grid)
lasso.coef = predict(out,type="coefficients",s=bestlam)[1:8,]
lasso.coef 
lasso.coef[lasso.coef!=0] 

# Modell für multiple lineare Regression
mlm <- lm(PM10 ~ AnzahlAutos+ Temperatur + Windgeschwindigkeit + Temperaturdifferenz + Windrichtung + Tagesstunde ,  data = dataset)
summary(mlm) 



