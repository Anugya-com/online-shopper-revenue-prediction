library(rpart)
library(rpart.plot)
library(caret)
install.packages("corrplot")
library(corrplot)


#Importing the Dataset

shop.df <- read.csv("online_shoppers_intention.csv")
View(shop.df)
#Basic Exploratory Data Analysis

#Understanding dimensions of the data set
dim(shop.df)
#First few values of the data set
head(shop.df) 
#Summary Statistics of the data set
summary(shop.df)
#Count of null values in the data frame column wise
sapply(shop.df, function(x) sum(is.na(x)))




shop.df['ProductRel_per_dur'] = shop.df['ProductRelated']/(shop.df['ProductRelated_Duration']+0.00001)
shop.df['Admin_per_dur'] = shop.df['Administrative']/(shop.df['Administrative_Duration']+0.00001)
shop.df['Inform_per_dur'] = shop.df['Informational']/(shop.df['Informational_Duration']+0.00001)
shop.df['Bounce_by_exit'] = shop.df['BounceRates']/(shop.df['ExitRates']+0.00001)

shop.df$Revenue <- factor(shop.df$Revenue, levels = c(FALSE, TRUE), 
                            labels = c("0", "1"))

new.shop <- shop.df[, -c(1:8)]  # delete columns 1 through 6 as they were highly correlated so we merged each of the two columns into single feature as shown above



ggplot(shop.df, aes(Month, ..count..)) + geom_bar(aes(fill = Revenue), position = "dodge")

View(new.shop)

#applying logistic regression


#partitioning the data 
# partition data


set.seed(3)
train.index <- sample(c(1:dim(new.shop)[1]), dim(new.shop)[1]*0.6)  
train.df <- new.shop[train.index, ]
valid.df <- new.shop[-train.index, ]


View(train.df)
View(valid.df)

# runnig logistic regression
# using glm() (general linear model) with family = "binomial" to fit a logistic regression.

logit.reg <- glm(Revenue ~ ., data = train.df, family = "binomial") 

options(scipen=999)

summary(logit.reg)



# using predict() with type = "response" to compute predicted probabilities. 
logit.reg.pred <- predict(logit.reg, valid.df, type = "response")

# first 5 actual and predicted records
data.frame(actual = valid.df$Revenue[1:5], predicted = logit.reg.pred[1:5])

logit.reg.pred.classes <- ifelse(logit.reg.pred > 0.5, 1, 0)
confusionMatrix(as.factor(logit.reg.pred.classes), as.factor(valid.df$Revenue))

# model selection
full.logit.reg <- glm(Revenue ~ ., data = train.df, family = "binomial") 
empty.logit.reg  <- glm(Revenue ~ 1,data = train.df, family= "binomial")
summary(empty.logit.reg)

backwards = step(full.logit.reg)
summary(backwards)


backwards.reg.pred <- predict(backwards, valid.df, type = "response")
backwards.reg.pred.classes <- ifelse(backwards.reg.pred > 0.5, 1, 0)
confusionMatrix(as.factor(backwards.reg.pred.classes), as.factor(valid.df$Revenue))



stepwise = step(empty.logit.reg,scope=list(lower=formula(empty.logit.reg),upper=formula(full.logit.reg)), direction="both",trace=1)
formula(stepwise)


final <- glm(Revenue ~ PageValues + ProductRel_per_dur + Month + SpecialDay + 
               Bounce_by_exit + Weekend + Inform_per_dur + VisitorType + 
               Admin_per_dur + OperatingSystems, data = train.df, family= "binomial")
summary(final)

final.logit.reg <- glm(Revenue ~ PageValues + ProductRel_per_dur + Month + SpecialDay + 
                   Bounce_by_exit + Weekend + Inform_per_dur + VisitorType + 
                   Admin_per_dur + OperatingSystems, data = train.df, family = "binomial") 

options(scipen=999)

summary(final.logit.reg)

final.logit.reg.pred <- predict(final.logit.reg, valid.df, type = "response")


data.frame(actual = valid.df$Revenue[1:5], predicted = final.logit.reg.pred[1:5])

final.logit.reg.pred.classes <- ifelse(final.logit.reg.pred > 0.5, 1, 0)
confusionMatrix(as.factor(final.logit.reg.pred.classes), as.factor(valid.df$Revenue))


