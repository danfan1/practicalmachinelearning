# Pracitcal Machine Learning Course Project
Ying Wai Fan  
3/11/2018  



## Objective

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.
In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
We will use the data to build a model to predict the corectness of barbell lifts from the accelerometer data.

## Data Source

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:
* exactly according to the specification (Class A)
* throwing the elbows to the front (Class B)
* lifting the dumbbell only halfway (Class C)
* lowering the dumbbell only halfway (Class D)
* throwing the hips to the front (Class E)

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

## Cleaning the Data

There is a lot of missing values (NA's) in the data set.
We only data fields with no missing values in all rows.
We remove fields, like timestamps, that have no connection to how well the exercise is performed.

I then retain 30% of the training set as validation, and only use the remaining 70% for model training.


```r
library(readr)
training <- read_csv("~/Desktop/pml-training.csv")
testing <- read_csv("~/Desktop/pml-testing.csv")

col.has.na <- apply(training, 2, function(x){any(is.na(x))})
training <- training[, !col.has.na]
testing <- testing[, !col.has.na]

training <- subset(training, select=-c(X1, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
testing <- subset(testing, select=-c(X1, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))

training$classe <- as.factor(training$classe)

library(caret)
inTrain <- createDataPartition(training$classe, p = 0.7, list=FALSE)
validation <- training[-inTrain,]
training <- training[inTrain,]
```

## Model 1: Decision Tree

I first try to build a model with decision tree.


```r
modelrpart <- train(classe ~ ., method="rpart", data=training)
predrpart <- predict(modelrpart,newdata=training)
sum(predrpart==training$classe) / length(predrpart)
```

```
## [1] 0.4957414
```

```r
table(predrpart, training$classe)
```

```
##          
## predrpart    A    B    C    D    E
##         A 3542 1098 1091 1016  360
##         B   64  907   74  404  352
##         C  287  653 1231  832  683
##         D    0    0    0    0    0
##         E   13    0    0    0 1130
```

The accuracy on the training set is only 0.5.
The confusion matrix also shows the model does separate the classes well.
What is worst is that it does not predict any case to class D.

I then apply the model on the validation set.
The accuracy and confusion matrix are simlar as those of the training set.


```r
predrpart.validation <- predict(modelrpart,newdata=validation)
sum(predrpart.validation==validation$classe) / length(predrpart.validation)
```

```
## [1] 0.4958369
```

```r
table(predrpart.validation, validation$classe)
```

```
##                     
## predrpart.validation    A    B    C    D    E
##                    A 1532  475  495  426  162
##                    B   26  389   35  174  141
##                    C  115  275  496  364  278
##                    D    0    0    0    0    0
##                    E    1    0    0    0  501
```

## Model 2: Random Forest

Then I try to build a model with random forest.


```r
library(randomForest)
modelrf <- randomForest(classe ~ ., data=training)
predrf <- predict(modelrf,newdata=training)
sum(predrf==training$classe) / length(predrf)
```

```
## [1] 1
```

```r
table(predrf, training$classe)
```

```
##       
## predrf    A    B    C    D    E
##      A 5580    0    0    0    0
##      B    0 3797    0    0    0
##      C    0    0 3422    0    0
##      D    0    0    0 3216    0
##      E    0    0    0    0 3607
```

The accuracy is 1, meaning the model can predict all cases in the training set correctly.
This is also shown by the confusion matrix.

I then apply the model on the validation set.
I also get perfect accuracy as on the training set.


```r
predrf.validation <- predict(modelrf,newdata=validation)
sum(predrf.validation==validation$classe) / length(predrf.validation)
```

```
## [1] 1
```

```r
table(predrf.validation, validation$classe)
```

```
##                  
## predrf.validation    A    B    C    D    E
##                 A 1674    0    0    0    0
##                 B    0 1139    0    0    0
##                 C    0    0 1026    0    0
##                 D    0    0    0  964    0
##                 E    0    0    0    0 1082
```


```r
plot(modelrf, main="Random Forest Model")
```

![](index_files/figure-html/plot-rf-1.png)<!-- -->

The error plot above shows that error drops quickly with just about 30 trees.
Random forest is quite efficient in reducing error in the prediction.

## Prediction on the Test Set

Now we apply the random forest model to the test set.


```r
predict(modelrf,newdata=testing)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
