Hopkins Machine Learning Project Report
================
Andy Hasselwander
February 16, 2018

Data Setup
==========

Check to see if data exist. If they do not, go and get them from the website. "allTraining" is subsequently split into training and testing sets. ""

    ## Loading required package: lattice

    ## Loading required package: ggplot2

Source: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: [http://groupware.les.inf.puc-rio.br/har\#wle\_paper\_section\#ixzz57ISl7FAA](http://groupware.les.inf.puc-rio.br/har#wle_paper_section#ixzz57ISl7FAA)

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

A: Exactly according to specifications B: Throwing the elbows to the front C: Lifting the dumbbell only halfway D: Lowering the dumbbell only halfway E: Throwing the hips to the front

Exploratory Analysis
====================

Not all of the variables have enough cases to create a valid training set. We eliminate all of the the variables that do not have complete cases in at least 90% of the rows. Not running "summary()" as a part of the Markdown to keep results reasonable.

``` r
library(ISLR); library(ggplot2); library(caret);
#summary(training)

#Create a vector of anything not a variable
ignoreVector <- c(1:7)

sum(complete.cases(training[,-ignoreVector]))
```

    ## [1] 277

``` r
#Too few cases to use only complete cases

sumCC <- function(x) {sum(complete.cases(x))}

ccVector <- apply(training[-ignoreVector], 2,  sumCC)
completeVars <- ccVector > (.9 * nrow(training))

training <- training[-ignoreVector]
training <- training[,completeVars]
```

Prediction: Decision Trees
==========================

Decision trees does not seem to work well. D is missed entirely. For all models, we us

``` r
require(rpart.plot)
```

    ## Loading required package: rpart.plot

    ## Loading required package: rpart

``` r
set.seed(66666)

if(!exists("decisionTreeModel")) {decisionTreeModel <- train(classe ~ ., data=training, method="rpart", preProcess="scale")}
rpart.plot(decisionTreeModel$finalModel)
```

![](course_project_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-3-1.png)

``` r
predictTree <- predict(decisionTreeModel, testing)

confusionMatrix(predictTree, testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1513  487  474  454  176
    ##          B   28  377   36  167  139
    ##          C  129  275  516  343  273
    ##          D    0    0    0    0    0
    ##          E    4    0    0    0  494
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4928          
    ##                  95% CI : (0.4799, 0.5056)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3364          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9038  0.33099  0.50292   0.0000  0.45656
    ## Specificity            0.6222  0.92204  0.79008   1.0000  0.99917
    ## Pos Pred Value         0.4874  0.50469  0.33594      NaN  0.99197
    ## Neg Pred Value         0.9421  0.85169  0.88273   0.8362  0.89085
    ## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
    ## Detection Rate         0.2571  0.06406  0.08768   0.0000  0.08394
    ## Detection Prevalence   0.5274  0.12693  0.26100   0.0000  0.08462
    ## Balanced Accuracy      0.7630  0.62652  0.64650   0.5000  0.72786

Prediction: Random Forests
==========================

Random Forest dramatically outperforms the decision tree model. For each class, the sensitivity and specificity are &gt;= 98%. The overall accuracy is 99.27%.

``` r
set.seed(66666)
if(!exists("rfModel")){
rfModel <- train(classe ~ ., data=training, method="rf", preProcess="scale")}

predictRF <- predict(rfModel, testing)

confusionMatrix(predictRF, testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1670   10    0    0    0
    ##          B    2 1125    3    1    0
    ##          C    1    4 1021   15    0
    ##          D    0    0    2  947    2
    ##          E    1    0    0    1 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9929          
    ##                  95% CI : (0.9904, 0.9949)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.991           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9976   0.9877   0.9951   0.9824   0.9982
    ## Specificity            0.9976   0.9987   0.9959   0.9992   0.9996
    ## Pos Pred Value         0.9940   0.9947   0.9808   0.9958   0.9982
    ## Neg Pred Value         0.9990   0.9971   0.9990   0.9966   0.9996
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2838   0.1912   0.1735   0.1609   0.1835
    ## Detection Prevalence   0.2855   0.1922   0.1769   0.1616   0.1839
    ## Balanced Accuracy      0.9976   0.9932   0.9955   0.9908   0.9989

``` r
testPredictRF <- predict(rfModel, validation)

#Predictions for the test set
data.frame(caseNumber = 1:20,predictions = testPredictRF)
```

    ##    caseNumber predictions
    ## 1           1           B
    ## 2           2           A
    ## 3           3           B
    ## 4           4           A
    ## 5           5           A
    ## 6           6           E
    ## 7           7           D
    ## 8           8           B
    ## 9           9           A
    ## 10         10           A
    ## 11         11           B
    ## 12         12           C
    ## 13         13           B
    ## 14         14           A
    ## 15         15           E
    ## 16         16           E
    ## 17         17           A
    ## 18         18           B
    ## 19         19           B
    ## 20         20           B

Prediction: Support Vector Machines
===================================

SVM performs better than decision tree, but not nearly as well as Random Forests.

``` r
set.seed(66666)

if(!exists("svmModel")) {svmModel <- train(classe ~ ., data=training, method="svmLinear", preProcess="scale")}

predictSVM <- predict(svmModel, testing)

confusionMatrix(predictSVM, testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1543  172   78   70   58
    ##          B   26  810   85   33  127
    ##          C   50   64  814  118   67
    ##          D   46   17   23  705   61
    ##          E    9   76   26   38  769
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.7886        
    ##                  95% CI : (0.778, 0.799)
    ##     No Information Rate : 0.2845        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.7311        
    ##  Mcnemar's Test P-Value : < 2.2e-16     
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9217   0.7112   0.7934   0.7313   0.7107
    ## Specificity            0.9102   0.9429   0.9385   0.9701   0.9690
    ## Pos Pred Value         0.8032   0.7493   0.7314   0.8275   0.8377
    ## Neg Pred Value         0.9670   0.9315   0.9556   0.9485   0.9370
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2622   0.1376   0.1383   0.1198   0.1307
    ## Detection Prevalence   0.3264   0.1837   0.1891   0.1448   0.1560
    ## Balanced Accuracy      0.9160   0.8270   0.8659   0.8507   0.8398
