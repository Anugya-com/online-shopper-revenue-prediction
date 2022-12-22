library(rpart)
library(rpart.plot)
library(caret)
install.packages("corrplot")
library(corrplot)


#Importing the Dataset

shop.df <- read.csv("online_shoppers_intention.csv")

#Basic Exploratory Data Analysis

#Understanding dimensions of the data set
dim(shop.df)
#First few values of the data set
head(shop.df) 
#Summary Statistics of the data set
summary(shop.df)
#Count of null values in the data frame column wise
sapply(shop.df, function(x) sum(is.na(x)))

#Encoding the categorical column

factors <- factor(shop.df$Month)
a <- print(as.numeric(factors))
shop.df$Month <- a

factors <- factor(shop.df$Weekend)
b <- print(as.numeric(factors))
shop.df$Weekend <- b


factors <- factor(shop.df$Revenue)
shop.df$Revenue <- print(as.numeric(factors))

factors <- factor(shop.df$VisitorType)
shop.df$VisitorType <- print(as.numeric(factors))


#Correlation Heat Map before feature analysis

Correlation <- cor(shop.df)
shop.df.cor = cor(shop.df, method = c("pearson"))


#From the above coorelation matrix, 
#it is clear that administrative data (both duration and point) are correlated. 
#Information, Product Related, and Rates(Exit and Bounce) have similar Characteristics. 
#Page Value seems to have a stronger correlation with the Revenue. 
#So To avoid the violation of the assumption, 
#we can combine the respective highly correlated features into a single feature.



shop.df['ProductRel_per_dur'] = shop.df['ProductRelated']/(shop.df['ProductRelated_Duration']+0.00001)
shop.df['Admin_per_dur'] = shop.df['Administrative']/(shop.df['Administrative_Duration']+0.00001)
shop.df['Inform_per_dur'] = shop.df['Informational']/(shop.df['Informational_Duration']+0.00001)
shop.df['Bounce_by_exit'] = shop.df['BounceRates']/(shop.df['ExitRates']+0.00001)



new.shop <- shop.df[, -c(1:8)]  # delete columns 1 through 8 as they were highly correlated so we merged each of the two columns into single feature as shown above

#Pearson correlation heat map after combining highly correlated features into one
Correlation <- cor(new.shop)
new.shop.cor = cor(new.shop, method = c("pearson"))
#Plotting  correlation matrix after combining highly correlated features into one
corrplot(new.shop.cor)

--------------------------------------------------------------------------------------------


# partitioning into training and test data
set.seed(1)  
train.index <- sample(c(1:dim(new.shop)[1]), dim(new.shop)[1]*0.6)  
train.df <- new.shop[train.index, ]
valid.df <- new.shop[-train.index, ]


--------------------------------------------------------------------------------------------
###Standard Decision Treee

##classification tree using genie index as default method
default.ct <- rpart(Revenue ~ ., data = train.df ,method = "class")

# plot tree
prp(default.ct, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = -15)
# count number of leaves
length(default.ct$frame$var[default.ct$frame$var == "<leaf>"])

#Checking model accuracy of the Default tree

#Step 1:find accuracy of the training dataset
default.ct.point.pred.train <- predict(default.ct,train.df,type = "class")
confusionMatrix(default.ct.point.pred.train, as.factor(train.df$Revenue))

#Step 2: Find accuracy with test dataset
default.ct.point.pred.valid <- predict(default.ct,valid.df,type = "class")
confusionMatrix(default.ct.point.pred.valid, as.factor(valid.df$Revenue))

-------------------------------------------------------------------------------------------
#Constructing a deeper tree to the point where misclassification rate of training data is 0.
deeper.ct <- rpart(Revenue ~ ., data = train.df, method = "class", cp = -1, minsplit = 1)
# count number of leaves
length(deeper.ct$frame$var[deeper.ct$frame$var == "<leaf>"])
# plot tree
prp(deeper.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -15,box.col=ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white')) 

#Checking model accuracy of the Deepest Tree

#Step 1:find accuracy of the training dataset
deeper.ct.point.pred.train <- predict(deeper.ct,train.df,type = "class")
confusionMatrix(deeper.ct.point.pred.train, as.factor(train.df$Revenue))

#Step 2: Find accuracy with test dataset
deeper.ct.point.pred.valid <- predict(deeper.ct,valid.df,type = "class")
confusionMatrix(deeper.ct.point.pred.valid, as.factor(valid.df$Revenue))

--------------------------------------------------------------------------------------

##Post-pruning to find the level at which the test dataset will find the highest accuracy
  
  
set.seed(1)
cv.ct <- rpart(Revenue ~ ., data = train.df, method = "class", cp = 0.00001, minsplit = 1, xval = 5)  # minsplit is the minimum number of observations in a node for a split to be attempted. xval is number K of folds in a K-fold cross-validation.
printcp(cv.ct)  # Print out the cp table of cross-validation errors. The R-squared for a regression tree is 1 minus rel error. xerror (or relative cross-validation error where "x" stands for "cross") is a scaled version of overall average of the 5 out-of-sample errors across the 5 folds.
pruned.ct <- prune(cv.ct, cp = 0.00344531)

printcp(pruned.ct)
prp(pruned.ct, type = 1, extra = 2, under = TRUE, split.font = 2, varlen = -20, 
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))   

#Checking model accuracy of the Pruned tree

#Step 1:find accuracy of the training dataset
pruned.ct.point.pred.train <- predict(pruned.ct,train.df,type = "class")
confusionMatrix(pruned.ct.point.pred.train, as.factor(train.df$Revenue))

#Step 2: Find accuracy with test dataset
pruned.ct.point.pred.valid <- predict(pruned.ct,valid.df,type = "class")
confusionMatrix(pruned.ct.point.pred.valid, as.factor(valid.df$Revenue))  
-------------------------------------------------------------------------------------
#miscellaneous
# classification tree using entropy/info gain as default method
default.info.ct <- rpart(Revenue ~ ., data = train.df, parms = list(split = 'information'), method = "class")
prp(default.info.ct, type = 1, extra = 2, under = TRUE, split.font = 2, varlen = -25)

#varlen variable length.. show complete name of the variable/node name

length(default.info.ct$frame$var[default.info.ct$frame$var == "<leaf>"])


--------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------

###Random Forest
  
library(randomForest)
## random forest
rf <- randomForest(as.factor(Revenue) ~ ., data = train.df, ntree = 500, 
                   mtry = 4, nodesize = 5, importance = TRUE)  

## variable importance plot
varImpPlot(rf, type = 1)


## confusion matrix
rf.pred.test <- predict(rf, valid.df)
rf.pred.train <- predict(rf, train.df)
confusionMatrix(rf.pred.test, as.factor(valid.df$Revenue))
confusionMatrix(rf.pred.train, as.factor(train.df$Revenue))


-------------------------------------------------------------------
rf <- randomForest(as.factor(Revenue) ~ ., data = train.df, ntree = 50, 
                     mtry = 4, nodesize = 5, importance = TRUE)  

## variable importance plot
varImpPlot(rf, type = 1)


## confusion matrix
rf.pred.test <- predict(rf, valid.df)
rf.pred.train <- predict(rf, train.df)
confusionMatrix(rf.pred.test, as.factor(valid.df$Revenue))
confusionMatrix(rf.pred.train, as.factor(train.df$Revenue))

---------------------------------------------------------------------------------
rf <- randomForest(as.factor(Revenue) ~ ., data = train.df, ntree = 10, 
                     mtry = 4, nodesize = 5, importance = TRUE)  

## variable importance plot
varImpPlot(rf, type = 1)


## confusion matrix
rf.pred.test <- predict(rf, valid.df)
confusionMatrix(rf.pred.test, as.factor(valid.df$Revenue))
rf.pred.train <- predict(rf, train.df)
confusionMatrix(rf.pred.train, as.factor(train.df$Revenue))

## Keep checking accuracy at different levels of number of trees


------------------------------------------------------------------------------------
  rf <- randomForest(as.factor(Revenue) ~ ., data = train.df, ntree = 30, 
                     mtry = 4, nodesize = 5, importance = TRUE)  

## variable importance plot
varImpPlot(rf, type = 1)


## confusion matrix
rf.pred.test <- predict(rf, valid.df)
confusionMatrix(rf.pred.test, as.factor(valid.df$Revenue))
rf.pred.train <- predict(rf, train.df)
confusionMatrix(rf.pred.train, as.factor(train.df$Revenue))

## Keep checking accuracy at different levels of number of trees

------------------------------------------------------------------------------------
rf <- randomForest(as.factor(Revenue) ~ ., data = train.df, ntree = 300, 
                     mtry = 4, nodesize = 5, importance = TRUE)  

## variable importance plot
varImpPlot(rf, type = 1)


## confusion matrix
rf.pred.test <- predict(rf, valid.df)
confusionMatrix(rf.pred.test, as.factor(valid.df$Revenue))
rf.pred.train <- predict(rf, train.df)
confusionMatrix(rf.pred.train, as.factor(train.df$Revenue))
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
###Boosted Trees
  
library(adabag)

train.df$Revenue <- as.factor(train.df$Revenue)
valid.df$Revenue <- as.factor(valid.df$Revenue)

set.seed(1)
boost <- boosting(Revenue ~ ., data = train.df)

pred <- predict(boost, valid.df)
confusionMatrix(as.factor(pred$class), as.factor(valid.df$Revenue))
pred.train.2 <- predict(boost, train.df)
confusionMatrix(as.factor(pred.train.2$class), as.factor(train.df$Revenue))

--------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------  
  
##miscellaneous
#exporting new.shop to excel for EDA  

install.packages("writexl")
library("writexl")
write_xlsx(new.shop, "C:\\Users\\AXC220053\\Desktop\\Fall'22\\Bus A with R\\Exam 2\\Project\\new_shop.xlsx")  


--------------------------------------------------------------------------------------