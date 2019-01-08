#Importing Dataset
library(RCurl)
x <- getURL("http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data")
spambase <- read.csv(text = x,header = FALSE)
attach(spambase)
count(spambase,'spam')

#Partitioning the data into 10 folds by first shuffling the data
#Randomly shuffle the data

spamData<-spambase[sample(nrow(spambase)),]
attach(spamData)

#Create 10 equally size folds

folds <- cut(seq(1,nrow(spamData)),breaks=10,labels=FALSE)

#Naive Bayes
install.packages("e1071")
library(e1071)
spambase$spam<-as.factor(spambase$spam)

spam_naive<-naiveBayes(spam~.,data = spambase)
spam_naive 
print(spam_naive)

#predicting the Naïve Bayes model using the entire dataset

NB_Predictions=predict(spam_naive,spambase,type ="class" )
table (NB_Predictions,spambase$spam, dnn = c("Predicted","Actual"))

#10-Fold Validation
CV_spam_naive <- lapply(1:10, function(x){ 
  model <-naiveBayes(spam~.,spamData[folds !=x, ])
  preds <- predict(model,  spamData[folds == x,], type="class")
  real <- spamData$spam[folds == x]
  conf <-confusionMatrix(preds, real)
  print(conf)
  return(data.frame(preds, real))
})

CV_spam_naive <- do.call(rbind, CV_spam_naive)
confusionMatrix(CV_spam_naive$preds, CV_spam_naive$real)

#Decision Tree Model-Pruned and final

control= rpart.control(minsplit = 10, minbucket = 10, maxdepth = 6, cp = 0.001)
dt_spam_train <- rpart(spam ~.,data=spamData ,method = "class",control = control,
                       parms=list(split = "information"))


Train_Accuracy = predict(dt_spam_train, newdata = spamData, type="class")
table(Train_Accuracy, spamData$spam, dnn = c("Predicted","Actual"))


#cross validation to our above tree

control= rpart.control(minsplit = 10, minbucket = 10 ,maxdepth = 6, cp = 0.001)
CV_spam_tree <- lapply(1:10, function(x){ 
  model <-rpart(spam ~.,data=spamData[folds != x,] ,method = "class",control = control,
                parms=list(split = "information"))
  preds <- predict(model,  spamData[folds == x,], type="class")
  real <- spamData$spam[folds == x]
  conf <-confusionMatrix(preds, real)
  print(conf)
  return(data.frame(preds, real))
})
CV_spam_tree <- do.call(rbind, CV_spam_tree)

confusionMatrix(CV_spam_tree$preds, CV_spam_tree$real)

#CV error Vs Tree Size plot
#plot the CV error by tree size

dt_spam_train$cptable
class(dt_spam_train$cptable)
cptable1<-data.frame(dt_spam_train$cptable)

library(ggplot2)
ggplot(cptable1, aes(x = cptable1$nsplit, y = cptable1$xerror)) +xlab("Tree size") +ylab("CV Error")+
  geom_line()+ geom_point(color = "red", size = 3) + ggtitle("CV Error Vs Tree Size") +
  theme_bw()

#Random Forest Model

rf <- randomForest(spam ~ ., data = spamData, ntree = 100, proximity = T, 
                   replace= T, importance = T, mtry = 3)

rf_Test_Accuracy = predict(rf, newdata = spamData, type = "class")
table(rf_Test_Accuracy, spamData$spam, dnn = c("Predicted", "Actual"))

#cross validation on Random Forest to report the testing error.

CV_spam_rf <- lapply(1:10, function(x){ 
  model <-randomForest(spam ~ ., data = spamData[folds != x,], ntree = 100, proximity = T, 
                       replace= T, importance = T, mtry = 3)
  preds <- predict(model,  spamData[folds == x,], type="class")
  real <- spamData$spam[folds == x]
  conf <-confusionMatrix(preds, real)
  print(conf)
  return(data.frame(preds, real))
})
CV_spam_rf <- do.call(rbind, CV_spam_rf)

confusionMatrix(CV_spam_rf$preds, CV_spam_rf$real)

#Importance of varaibles
varImpPlot(rf)

#AdaBoost

install.packages("adabag")
library(adabag)
spam_adaboost <- boosting(spam ~., data = spamData, mfinal = 100)

boost_error <- predict.boosting(spam_adaboost, newdata = spamData)

#Cross Validation 

CV_spam_boost <- lapply(1:10, function(x){ 
  model <-boosting(spam ~., data = spamData[folds != x,], mfinal = 100)
  preds <- predict.boosting(model, newdata =  spamData[folds == x,])
  real <- spamData$spam[folds == x]
  #conf <-confusionMatrix(preds, real)
  print(preds$confusion)
  return(data.frame(preds$class, real))
})
CV_spam_boost <- do.call(rbind, CV_spam_boost)
confusionMatrix(CV_spam_boost$preds, CV_spam_boost$real)

#Plotting importance of variables

par(mar=c(12,4,4,4))
barplot(spam_adaboost$imp[order(spam_adaboost$imp, decreasing = TRUE)], 
        ylim = c(0, 10), main = "Variables Relative Importance",
        col = "lightblue", las=2)

#ROC

#Naïve Bayes
model <-naiveBayes(spam~.,spamData[folds !=1, ])
#RUN PREDICTION AND GET PROBABILITIES
pred <- predict(model,  spamData[folds == 1,], type = "raw")
pr <- prediction(pred[,2],spamData[folds == 1,]$spam)
perf<-performance(pr,"tpr", "fpr")
plot(perf)


#Random Forest

model <-randomForest(spam ~ ., data = spamData[folds != 1,], ntree = 100, proximity = T, 
                     replace= T, importance = T, mtry = 3)
pred <- predict(model,  spamData[folds == 1,], type= "prob")
pr <- prediction(pred[,2],spamData[folds == 1,]$spam)
perf<-performance(pr,"tpr", "fpr")
plot(perf)

#Decision Tree
control= rpart.control(minsplit = 10, minbucket = 10, maxdepth = 6, cp = 0.001)
model <-rpart(spam ~.,data=spamData[folds != 1,] ,method = "class",control = control,
              parms=list(split = "information"))
pred <- predict(model,  spamData[folds == 1,], type="prob")
pr <- prediction(pred[,2],spamData[folds == 1,]$spam)
perf<-performance(pr,"tpr", "fpr")
plot(perf)





