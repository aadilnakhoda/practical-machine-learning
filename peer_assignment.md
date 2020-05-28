Executive Summary
=================

With the advent of fitness devices and apps such as *Jawbone Up*, *Nike Fuelband* and *Fitbit*, it is now possible to measure not only how much of an activity people are doing but how well they are doing those activities.This project uses data on accelerometer on belt, forearm, arm and dumbell of 6 participants to predict the manner in which they do the exercise. The following analysis makes use of machine learning models such as random forest, gradient boosting and recursive partitioning and regression trees. The results from the random forest are the most accurate.

Loading the data
----------------

The data files available on the course webpage are downloaded into the appropriate directory and then called from that location. **The code is reported at the end of this report**.

Data Cleaning and Training
--------------------------

First, the variables containing NA and missing values have been replaced with *NA* in the dataset. The columns containing *NA* variables are then removed from the dataset.

    ## 'data.frame':    19622 obs. of  60 variables:
    ##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ user_name           : chr  "carlitos" "carlitos" "carlitos" "carlitos" ...
    ##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
    ##  $ cvtd_timestamp      : chr  "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" ...
    ##  $ new_window          : chr  "no" "no" "no" "no" ...
    ##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
    ##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
    ##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
    ##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
    ##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
    ##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
    ##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
    ##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
    ##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
    ##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
    ##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
    ##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
    ##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
    ##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
    ##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
    ##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
    ##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
    ##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
    ##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
    ##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
    ##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
    ##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
    ##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
    ##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
    ##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
    ##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
    ##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
    ##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
    ##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
    ##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
    ##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
    ##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
    ##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
    ##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
    ##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
    ##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
    ##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
    ##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
    ##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
    ##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
    ##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
    ##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
    ##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
    ##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
    ##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
    ##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
    ##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
    ##  $ classe              : chr  "A" "A" "A" "A" ...

    ## [1] 19622    60

    ## [1] 20 60

Training and testing the model
------------------------------

We will divide our dataset into two portions, one that will be used to build our respective machine learning models and the other used to determine the accuracy of our models. We will allocate 70 percent to the training portion and the remaining 30 percent to the testing portion.

Dimension and the variable names of the training set. The first seven variables listed in the relevant dataset are removed.

    ## [1] 13737    53

    ##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
    ## 1      1.41       8.07    -94.4                3         0.00         0.00
    ## 2      1.41       8.07    -94.4                3         0.02         0.00
    ## 3      1.42       8.07    -94.4                3         0.00         0.00
    ## 5      1.48       8.07    -94.4                3         0.02         0.02
    ## 7      1.42       8.09    -94.4                3         0.02         0.00
    ## 8      1.42       8.13    -94.4                3         0.02         0.00
    ##   gyros_belt_z accel_belt_x accel_belt_y accel_belt_z magnet_belt_x
    ## 1        -0.02          -21            4           22            -3
    ## 2        -0.02          -22            4           22            -7
    ## 3        -0.02          -20            5           23            -2
    ## 5        -0.02          -21            2           24            -6
    ## 7        -0.02          -22            3           21            -4
    ## 8        -0.02          -22            4           21            -2
    ##   magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm
    ## 1           599          -313     -128      22.5    -161              34
    ## 2           608          -311     -128      22.5    -161              34
    ## 3           600          -305     -128      22.5    -161              34
    ## 5           600          -302     -128      22.1    -161              34
    ## 7           599          -311     -128      21.9    -161              34
    ## 8           603          -313     -128      21.8    -161              34
    ##   gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
    ## 1        0.00        0.00       -0.02        -288         109        -123
    ## 2        0.02       -0.02       -0.02        -290         110        -125
    ## 3        0.02       -0.02       -0.02        -289         110        -126
    ## 5        0.00       -0.03        0.00        -289         111        -123
    ## 7        0.00       -0.03        0.00        -289         111        -125
    ## 8        0.02       -0.02        0.00        -289         111        -124
    ##   magnet_arm_x magnet_arm_y magnet_arm_z roll_dumbbell pitch_dumbbell
    ## 1         -368          337          516      13.05217      -70.49400
    ## 2         -369          337          513      13.13074      -70.63751
    ## 3         -368          344          513      12.85075      -70.27812
    ## 5         -374          337          506      13.37872      -70.42856
    ## 7         -373          336          509      13.12695      -70.24757
    ## 8         -372          338          510      12.75083      -70.34768
    ##   yaw_dumbbell total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y
    ## 1    -84.87394                   37                0            -0.02
    ## 2    -84.71065                   37                0            -0.02
    ## 3    -85.14078                   37                0            -0.02
    ## 5    -84.85306                   37                0            -0.02
    ## 7    -85.09961                   37                0            -0.02
    ## 8    -85.09708                   37                0            -0.02
    ##   gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
    ## 1                0             -234               47             -271
    ## 2                0             -233               47             -269
    ## 3                0             -232               46             -270
    ## 5                0             -233               48             -270
    ## 7                0             -232               47             -270
    ## 8                0             -234               46             -272
    ##   magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
    ## 1              -559               293               -65         28.4
    ## 2              -555               296               -64         28.3
    ## 3              -561               298               -63         28.3
    ## 5              -554               292               -68         28.0
    ## 7              -551               295               -70         27.9
    ## 8              -555               300               -74         27.8
    ##   pitch_forearm yaw_forearm total_accel_forearm gyros_forearm_x gyros_forearm_y
    ## 1         -63.9        -153                  36            0.03            0.00
    ## 2         -63.9        -153                  36            0.02            0.00
    ## 3         -63.9        -152                  36            0.03           -0.02
    ## 5         -63.9        -152                  36            0.02            0.00
    ## 7         -63.9        -152                  36            0.02            0.00
    ## 8         -63.8        -152                  36            0.02           -0.02
    ##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
    ## 1           -0.02             192             203            -215
    ## 2           -0.02             192             203            -216
    ## 3            0.00             196             204            -213
    ## 5           -0.02             189             206            -214
    ## 7           -0.02             195             205            -215
    ## 8            0.00             193             205            -213
    ##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
    ## 1              -17              654              476      A
    ## 2              -18              661              473      A
    ## 3              -18              658              469      A
    ## 5              -17              655              473      A
    ## 7              -18              659              470      A
    ## 8               -9              660              474      A

Correlation Matrix
------------------

A correlation matrix is plotted to determine the most correlated features. Positive correlations are plotted in blue color and the negative correlations are plotted in red color. In other words, the plot will help show the relationship that exists between different variables/features. The correlation uses the first principal component (FPC) order. ![](peer_assignment_files/figure-markdown_github/unnamed-chunk-7-1.png)

The variables or features that report a correlation with another variable/feature of at least 0.7 are presented below.

    ##  [1] "accel_belt_z"      "roll_belt"         "accel_arm_y"      
    ##  [4] "accel_belt_y"      "yaw_belt"          "accel_dumbbell_z" 
    ##  [7] "accel_belt_x"      "pitch_belt"        "magnet_dumbbell_x"
    ## [10] "accel_dumbbell_y"  "magnet_dumbbell_y" "accel_dumbbell_x" 
    ## [13] "accel_arm_x"       "accel_arm_z"       "magnet_arm_y"     
    ## [16] "magnet_belt_z"     "accel_forearm_y"   "gyros_arm_x"      
    ## [19] "gyros_forearm_y"

The Machine Learning Models
---------------------------

We will use the following machine learning algorithms:

1.  Decision Trees using rpart package
2.  Gradient Boosting Algorithm using GBM method
3.  Random Forest The random forest provides us with the greatest accuracy. The following models include the respective confusion matrices that highlight accuracy and the value of kappa. The train control uses the cross-validation method with 5 number of folds or number of resampling iterations.

### Decision Trees

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1508  458  459  442  143
    ##          B   36  400   33  162  141
    ##          C  122  281  534  360  285
    ##          D    0    0    0    0    0
    ##          E    8    0    0    0  513
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.5021         
    ##                  95% CI : (0.4893, 0.515)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.35           
    ##                                          
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9008  0.35119  0.52047   0.0000  0.47412
    ## Specificity            0.6433  0.92162  0.78432   1.0000  0.99833
    ## Pos Pred Value         0.5010  0.51813  0.33755      NaN  0.98464
    ## Neg Pred Value         0.9423  0.85547  0.88566   0.8362  0.89392
    ## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
    ## Detection Rate         0.2562  0.06797  0.09074   0.0000  0.08717
    ## Detection Prevalence   0.5115  0.13118  0.26882   0.0000  0.08853
    ## Balanced Accuracy      0.7721  0.63640  0.65239   0.5000  0.73623

![](peer_assignment_files/figure-markdown_github/unnamed-chunk-9-1.png)

The numbers shown in the nodes are the predicted values for each class at a particular activity. The decision tree has not only a poor accuracy but fails to identify D as well as gives the highest probability to A. The following results should provide us with greater evidence.

### Gradient Boosting Algorithms

Boosting provides a 'boost' to the weak learners to attain the level of performance of more powerful learners. Each additional cluster in this method should perform better than by random chance. As the number of boosting iterations increases, the accuracy increases. The same can be said for the max tree depth shown as three separate lines.

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2320
    ##      2        1.4608             nan     0.1000    0.1598
    ##      3        1.3594             nan     0.1000    0.1301
    ##      4        1.2766             nan     0.1000    0.1127
    ##      5        1.2081             nan     0.1000    0.0830
    ##      6        1.1552             nan     0.1000    0.0754
    ##      7        1.1074             nan     0.1000    0.0758
    ##      8        1.0609             nan     0.1000    0.0633
    ##      9        1.0214             nan     0.1000    0.0532
    ##     10        0.9876             nan     0.1000    0.0471
    ##     20        0.7608             nan     0.1000    0.0268
    ##     40        0.5313             nan     0.1000    0.0117
    ##     60        0.4056             nan     0.1000    0.0091
    ##     80        0.3234             nan     0.1000    0.0041
    ##    100        0.2699             nan     0.1000    0.0058
    ##    120        0.2252             nan     0.1000    0.0024
    ##    140        0.1923             nan     0.1000    0.0016
    ##    150        0.1788             nan     0.1000    0.0014

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1651   26    0    0    1
    ##          B   20 1074   32    3   24
    ##          C    1   35  983   34    8
    ##          D    0    3    9  920   11
    ##          E    2    1    2    7 1038
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9628          
    ##                  95% CI : (0.9576, 0.9675)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9529          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9863   0.9429   0.9581   0.9544   0.9593
    ## Specificity            0.9936   0.9834   0.9839   0.9953   0.9975
    ## Pos Pred Value         0.9839   0.9315   0.9265   0.9756   0.9886
    ## Neg Pred Value         0.9945   0.9863   0.9911   0.9911   0.9909
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2805   0.1825   0.1670   0.1563   0.1764
    ## Detection Prevalence   0.2851   0.1959   0.1803   0.1602   0.1784
    ## Balanced Accuracy      0.9899   0.9631   0.9710   0.9748   0.9784

![](peer_assignment_files/figure-markdown_github/unnamed-chunk-10-1.png)

### Random Forest

Random forest is often useful when there are a large number of features. This method works well as it selects only the most important of all features and is less prone to overfitting. As the number of features increases beyond 28 (approximately), the accuracy falls.

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    5    0    0    0
    ##          B    2 1131   15    0    0
    ##          C    0    3 1011   13    0
    ##          D    0    0    0  951    2
    ##          E    0    0    0    0 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9932          
    ##                  95% CI : (0.9908, 0.9951)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9914          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9930   0.9854   0.9865   0.9982
    ## Specificity            0.9988   0.9964   0.9967   0.9996   1.0000
    ## Pos Pred Value         0.9970   0.9852   0.9844   0.9979   1.0000
    ## Neg Pred Value         0.9995   0.9983   0.9969   0.9974   0.9996
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1922   0.1718   0.1616   0.1835
    ## Detection Prevalence   0.2850   0.1951   0.1745   0.1619   0.1835
    ## Balanced Accuracy      0.9988   0.9947   0.9910   0.9931   0.9991

![](peer_assignment_files/figure-markdown_github/unnamed-chunk-11-1.png)

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                      Overall
    ## roll_belt             100.00
    ## yaw_belt               78.98
    ## magnet_dumbbell_z      67.89
    ## pitch_belt             61.70
    ## magnet_dumbbell_y      59.62
    ## pitch_forearm          57.21
    ## magnet_dumbbell_x      53.28
    ## roll_forearm           52.44
    ## magnet_belt_y          44.55
    ## accel_belt_z           44.51
    ## accel_dumbbell_y       44.25
    ## magnet_belt_z          42.88
    ## roll_dumbbell          42.22
    ## accel_dumbbell_z       38.34
    ## roll_arm               35.66
    ## accel_forearm_x        35.24
    ## yaw_dumbbell           31.25
    ## accel_arm_x            29.05
    ## accel_dumbbell_x       28.95
    ## total_accel_dumbbell   28.27

Report the Accuracy and the Out-of-Sample Error
-----------------------------------------------

The accuracy of each of the algorithm is summarized in the following box and whisker plot. It is observed that the random forest algorithm has the highest level of accuracy. The kappa statistics tells us how much better is the chosen classifier in performance compared to simple guesses at random. Random forest reports the highest value for kappa as well. ![](peer_assignment_files/figure-markdown_github/unnamed-chunk-13-1.png)

The accuracy for RPART is 0.502124 and the out-of-sample error is 0.497876.The accuracy for GBM is 0.9627867 and the out-of-sample error is 0.0372133. The accuracy for random forest is 0.9932031 and the out-of-sample error is 0.0067969. Therefore, we prefer the random forest as the appropriate algorithm.

Conclusion.
-----------

The random forest machine learning algorithm results in 0.9932031 accuracy and 0.0067969 out-of-sample error. Both the acurracy and kappa favor the random forest algorithm.

Results for the Prediction Quiz.
================================

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Code
----

``` r
#Include the following libraries
library(caret)
library(randomForest)
library(e1071)
library(parallel)
library(doParallel)
library(rattle)
#To ensure that multiple cores are used to reduce processing time
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
# The verson: 3.5.1 is used for the random number generator. This is consistent with the quizzes attempted in the course.  
RNGkind()
RNGversion("3.5.1")
set.seed(12345678)
#Downloading the file, saving into appropriate location
setwd("E:/Coursera/Practical Machine Learning/Week 4/Assignment/")
if(!file.exists("./data")){dir.create("./data")
trainingURL <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
download.file(trainingURL,destfile="./data/pml-training.csv")
projquizURL <-'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
download.file(projquizURL,destfile="./data/pml-testing.csv")}
# Read the files for the main peer assessment exercise (training) and the supplement project prediction quiz (projquiz) 
training <- read.csv("./data/pml-training.csv", na.strings=c("NA", ""), stringsAsFactors = F, header=T)

projquiz <- read.csv("./data/pml-testing.csv", na.strings=c("NA", ""), stringsAsFactors = F, header=T)


# Remove the colums containing NA observations
training<- training[, colSums(is.na(training)) == 0]
projquiz<- projquiz[, colSums(is.na(projquiz)) == 0]
# The dimensions of the two datasets created for the machine learning exercise 
str(training)
dim(training)
dim(projquiz)
# Remove the identifying variables. These are not to be included in the prediction algorithms.  
training <- training[,-c(1:7)]
projquiz <- projquiz[,-c(1:7)]
#Creating the data partitions. The training data will be 70% and the testing data will be 30%. 
inTrain <- createDataPartition(training$classe, p=0.7, list=F)
trainset <- training[inTrain, ]
testset <-  training[-inTrain,]
dim(trainset)
head(trainset)

# Creating the correlation matrix
library(corrplot)
correlation_mat <- cor(trainset[, -53])
par(mar=c(5.1,4.1,5,2.1))
corrplot(correlation_mat, order="FPC", method="color", type="lower", tl.cex=0.7, tl.col=rgb(0,0,0))
correlated <- findCorrelation(correlation_mat, cutoff=0.7)
names(trainset)[correlated]

# RPART Decision Tree Algorithm
fitControl <- trainControl(method = "cv", number = 5, verboseIter = F, allowParallel = T)
model_rpart <- train(as.factor(classe)~., data=trainset, method="rpart", trControl=fitControl)
predict_rpart <- predict(model_rpart, newdata=testset)
conf_rpart <- confusionMatrix(predict_rpart, as.factor(testset$classe))
conf_rpart
fancyRpartPlot(model_rpart$finalModel, main="Decision Tree from RPART Algorithm")
# Gradient Boosting Method (GBM) Algorithm
fitControl <- trainControl(method = "cv", number = 5, verboseIter = F, allowParallel = T)
model_gbm <- train(as.factor(classe)~., data=trainset, method="gbm", trControl=fitControl)
predict_gbm <- predict(model_gbm, newdata=testset)
conf_gbm <- confusionMatrix(predict_gbm, as.factor(testset$classe))
conf_gbm
plot(model_gbm, main="Accuracy of Gradient Boosting Method")
# Random Forest Algorithm
fitControl <- trainControl(method = "cv", number = 5, verboseIter = F, allowParallel = T)
model_rf <- train(as.factor(classe)~., data=trainset, method="rf", trControl=fitControl)
stopCluster(cluster)
registerDoSEQ()
predict_rf <- predict(model_rf, newdata=testset)
conf_rf <- confusionMatrix(predict_rf, as.factor(testset$classe))
conf_rf
plot(model_rf, main="Accuracy of Random Forest Algorithm")
# Important features
important_features <- varImp(model_rf)
# Twenty most important features
important_features

# Reporting the accuracy of the three algorithms

conf_rpart$overall['Accuracy']
conf_rf$overall['Accuracy']
conf_gbm$overall['Accuracy']

# Presenting the accuracy of the three algorithms in box and whisker plot
results <- resamples((list(RPART=model_rpart, GBM=model_gbm, RF=model_rf)))
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales, main="Accuracy and Kappa Values for each Model")
# Reporting the out-of-sample error of the three algorithms
accuracy_rpart <- sum(predict_rpart ==testset$classe)/length(predict_rpart)
outofsampleerror_rpart <- 1-accuracy_rpart
accuracy_gbm <- sum(predict_gbm ==testset$classe)/length(predict_gbm)
outofsampleerror_gbm <- 1-accuracy_gbm
accuracy_rf <- sum(predict_rf ==testset$classe)/length(predict_rf)
outofsampleerror_rf <- 1-accuracy_rf
# The answers to the poject prediction quiz portion. 
predict_quiz <- predict(model_rf, newdata=projquiz)
predict_quiz
```
