# Single layer neural network

This example using the ISOLET data illustrates the use of simple
neural networks (NNs), and also highlights some issues of which it may 
be important to be aware. As we discussed in class, NNs typically have 
more parameters than observations and a number of tuning parameters
that need to be chosen by the user. Among these: the number of 
hidden layers, the number of units in each layer, the *activation function*,
the *loss function*, a decaying factor, and the initial point 
at which to start the optimization iterations. In the example below we illustrate 
some difficulties that can be encountered when trying to find 
which tuning parameters to use to train a NN.

In order to focus on the concepts behind NN, we will use the `nnet` 
package in `R`. This package is a very simple implementation 
of NNs with a single hidden layer, and relies on standard optimization
algorithms to train it. Such simple setting will allow us to 
separate implementation / optimization issues from the underlying
model and ideas behind NN, which carry over naturally to more
complex NNs. 

For our example we will use again the ISOLET data which is available here: [http://archive.ics.uci.edu/ml/datasets/ISOLET](http://archive.ics.uci.edu/ml/datasets/ISOLET), along with more information about it. It contains data on sound recordings of 150 speakers saying each letter of the alphabet (twice). See the original source for more details. The full data file is rather large and available in compressed form. 
Instead, we will read it from a private copy in plain text form I made 
available on Dropbox.  

## "C" and "Z"
First we look at building a classifier to identify the letters C and Z. This 
is the simplest scenario and it will help us fix ideas. We now read the 
full data set, and extract the training and test rows corresponding to those
two letters:

```r
library(nnet)
xx.tr <- readRDS("data/isolet-train.RDS")
xx.te <- readRDS("data/isolet-test.RDS")
lets <- c(3, 26)
LETTERS[lets]
#> [1] "C" "Z"
# Training set
x.tr <- xx.tr[xx.tr$V618 %in% lets, ]
x.tr$V618 <- as.factor(x.tr$V618)
# Test set
x.te <- xx.te[xx.te$V618 %in% lets, ]
truth <- x.te$V618 <- as.factor(x.te$V618)
```
We train a NN with a single hidden layer, and a single unit in the hidden layer. 

```r
set.seed(123)
a1 <- nnet(V618 ~ ., data = x.tr, size = 1, decay = 0, maxit = 1500, MaxNWts = 2000)
#> # weights:  620
#> initial  value 350.425020 
#> iter  10 value 41.176789
#> iter  20 value 18.095256
#> iter  30 value 18.052107
#> iter  40 value 18.050646
#> iter  50 value 18.050036
#> iter  60 value 18.048042
#> iter  70 value 12.957465
#> iter  80 value 6.911704
#> iter  90 value 6.483384
#> iter 100 value 6.482800
#> iter 110 value 6.482764
#> iter 120 value 6.482733
#> final  value 6.482722 
#> converged
```
Note the slow convergence. The final value of the objective value was:

```r
a1$value
#> [1] 6.482722
```
The error rate on the training set ("goodness of fit") is

```r
b1 <- predict(a1, type = "class") # , type='raw')
mean(b1 != x.tr$V618)
#> [1] 0.002083333
```
We see that this NN fits the training set perfectly. Is this desirable? 

We now run the algorithm again, with a different starting point. 

```r
set.seed(456)
a2 <- nnet(V618 ~ ., data = x.tr, size = 1, decay = 0, maxit = 1500, MaxNWts = 2000)
#> # weights:  620
#> initial  value 336.934868 
#> iter  10 value 157.630462
#> iter  20 value 61.525474
#> iter  30 value 48.367801
#> iter  40 value 42.822797
#> iter  50 value 36.541767
#> iter  60 value 36.476799
#> iter  70 value 33.120685
#> iter  80 value 26.860343
#> iter  90 value 26.859999
#> final  value 26.859999 
#> converged
```
Compare
the attained value of the objective and the error rate on the training set
with those above (6.482722 and 0, respectively):

```r
a2$value
#> [1] 26.86
b2 <- predict(a2, type = "class") # , type='raw')
mean(b2 != x.tr$V618)
#> [1] 0.01041667
```
So, we see that the second run of NN produces a much worse solution.
How are their performances on the test set?

```r
b1 <- predict(a1, newdata = x.te, type = "class") # , type='raw')
mean(b1 != x.te$V618)
#> [1] 0.03333333
b2 <- predict(a2, newdata = x.te, type = "class") # , type='raw')
mean(b2 != x.te$V618)
#> [1] 0.04166667
```
The second (worse) solution performs better on the test set. 

What if we add more units to the hidden layer? We increase the
number of units on the hidden layer from 3 to 6. 

```r
set.seed(123)
a1 <- nnet(V618 ~ ., data = x.tr, size = 3, decay = 0, maxit = 1500, MaxNWts = 2000, trace = FALSE)
set.seed(456)
a2 <- nnet(V618 ~ ., data = x.tr, size = 3, decay = 0, maxit = 1500, MaxNWts = 2000, trace = FALSE)
```
The objective functions are 

```r
a1$value
#> [1] 6.482738
a2$value
#> [1] 9.052402e-05
```
respectively, and their performance on the training and test sets are:

```r
b1 <- predict(a1, type = "class") # , type='raw')
mean(b1 != x.tr$V618)
#> [1] 0.002083333
b2 <- predict(a2, type = "class") # , type='raw')
mean(b2 != x.tr$V618)
#> [1] 0

b1 <- predict(a1, newdata = x.te, type = "class") # , type='raw')
mean(b1 != x.te$V618)
#> [1] 0.03333333
b2 <- predict(a2, newdata = x.te, type = "class") # , type='raw')
mean(b2 != x.te$V618)
#> [1] 0.04166667
```
Again we note that the (seemingly much) worse solution (in terms of the objective
function whose optimization defines the NN) performs better 
on the test set. 

What if we add a decaying factor as a form of regularization? 

```r
set.seed(123)
a1 <- nnet(V618 ~ ., data = x.tr, size = 3, decay = 0.05, maxit = 500, MaxNWts = 2000, trace = FALSE)
a1$value
#> [1] 5.345279
set.seed(456)
a2 <- nnet(V618 ~ ., data = x.tr, size = 3, decay = 0.05, maxit = 500, MaxNWts = 2000, trace = FALSE)
a2$value
#> [1] 5.345279
```
Now the two solutions starting from these random initial values 
are the same (the reader is encouraged to 
try more random starts). How does this NN do on the training and test sets?

```r
b1 <- predict(a1, type = "class") # , type='raw')
mean(b1 != x.tr$V618)
#> [1] 0
b1 <- predict(a1, newdata = x.te, type = "class") # , type='raw')
mean(b1 != x.te$V618)
#> [1] 0.008333333
```

Note that this "regularized" solution which corresponds to a 
slightly better solution than the worse one above in terms
of objective function (but still much worse than the best ones)
performs noticeably better on the test set. This seem to suggest
that it is not easy to select which of the many local extrema to used
based  on the objective function values they attain. 

Another tuning parameter we can vary is the number of units
in the hidden layer, which will also increase significantly the
number of possible weight parameters in our model. 
The above solution uses 1858 weights. We now add more 
units to the hidden layer (6 instead of 3) and increase the limit on
the number of allowable weights to 4000: 

```r
set.seed(123)
a1 <- nnet(V618 ~ ., data = x.tr, size = 6, decay = 0.05, maxit = 500, MaxNWts = 4000, trace = FALSE)
a1$value
#> [1] 4.777806
set.seed(456)
a2 <- nnet(V618 ~ ., data = x.tr, size = 6, decay = 0.05, maxit = 500, MaxNWts = 4000, trace = FALSE)
a2$value
#> [1] 4.172023
b1 <- predict(a1, type = "class") # , type='raw')
mean(b1 != x.tr$V618)
#> [1] 0

b2 <- predict(a2, type = "class") # , type='raw')
mean(b2 != x.tr$V618)
#> [1] 0

b1 <- predict(a1, newdata = x.te, type = "class") # , type='raw')
mean(b1 != x.te$V618)
#> [1] 0.008333333

b2 <- predict(a2, newdata = x.te, type = "class") # , type='raw')
mean(b2 != x.te$V618)
#> [1] 0.008333333
```
Note that both of these two distinct solutions fit the training set 
exactly (0 apparent error rate), and have the same performance
on the test set. We leave it to the reader to perform a more
exhaustive study of the prediction properties of these solutions
using an appropriate CV experiment. 

## More letters

We now repeat the same exercise above but on a 4-class
setting. 

```r
lets <- c(3, 7, 9, 26)
x.tr <- xx.tr[xx.tr$V618 %in% lets, ]
x.tr$V618 <- as.factor(x.tr$V618)
# testing set
x.te <- xx.te[xx.te$V618 %in% lets, ]
truth <- x.te$V618 <- as.factor(x.te$V618)
```

The following tries show that a NN with 
only one unit in the hidden layer does not perform well.
As before, we compare two local minima of the NN training
algorithm. First we show the values of the
corresponding local minima of the objective function, and then
their error rates on the training and test sets.

```r
set.seed(123)
a1 <- nnet(V618 ~ ., data = x.tr, size = 1, decay = 0, maxit = 1500, MaxNWts = 2000, trace = FALSE)
a1$value
#> [1] 9.711177e-05
set.seed(456)
a2 <- nnet(V618 ~ ., data = x.tr, size = 1, decay = 0, maxit = 1500, MaxNWts = 2000, trace = FALSE)
a2$value
#> [1] 789.9009
b1 <- predict(a1, type = "class") # , type='raw')
mean(b1 != x.tr$V618)
#> [1] 0
b2 <- predict(a2, type = "class") # , type='raw')
mean(b2 != x.tr$V618)
#> [1] 0.4875
b1 <- predict(a1, newdata = x.te, type = "class") # , type='raw')
mean(b1 != x.te$V618)
#> [1] 0.4625
b2 <- predict(a2, newdata = x.te, type = "class") # , type='raw')
mean(b2 != x.te$V618)
#> [1] 0.5375
```
Note that the error rates on the test set are
0.462 and 
0.537, which are
very high.
Better results are obtained with 6 units on the hidden layer
and a slightly regularized solution. As before, 
use two runs of the training
algorithm and look at the corresponding values of the
objective function, and the error rates 
of both NNs on the training and test sets.

```r
set.seed(123)
a1 <- nnet(V618 ~ ., data = x.tr, size = 6, decay = 0.05, maxit = 500, MaxNWts = 4000, trace = FALSE)
a1$value
#> [1] 9.037808
set.seed(456)
a2 <- nnet(V618 ~ ., data = x.tr, size = 6, decay = 0.05, maxit = 500, MaxNWts = 4000, trace = FALSE)
a2$value
#> [1] 9.171046
b1 <- predict(a1, type = "class") # , type='raw')
mean(b1 != x.tr$V618)
#> [1] 0
b2 <- predict(a2, type = "class") # , type='raw')
mean(b2 != x.tr$V618)
#> [1] 0
b1 <- predict(a1, newdata = x.te, type = "class") # , type='raw')
mean(b1 != x.te$V618)
#> [1] 0.0125
b2 <- predict(a2, newdata = x.te, type = "class") # , type='raw')
mean(b2 != x.te$V618)
#> [1] 0.0125
```
The error rates on the test set are now 
0.013 and 
0.013, which are
much better than before.

## Even more letters

We now consider building a classifier with 7 classes, which 
is a more challenging problem. 

```r
lets <- c(3, 5, 7, 9, 12, 13, 26)
LETTERS[lets]
#> [1] "C" "E" "G" "I" "L" "M" "Z"
x.tr <- xx.tr[xx.tr$V618 %in% lets, ]
x.tr$V618 <- as.factor(x.tr$V618)
# testing set
x.te <- xx.te[xx.te$V618 %in% lets, ]
truth <- x.te$V618 <- as.factor(x.te$V618)
```
The following code trains a NN with 6 units on the hidden layer and
moderate regularization (via a decaying factor of `0.3` and 
an upper limit of 4000 weights).

```r
set.seed(123)
a1 <- nnet(V618 ~ ., data = x.tr, size = 6, decay = 0.3, maxit = 1500, MaxNWts = 4000, trace = FALSE)
a1$value
#> [1] 102.1805
set.seed(456)
a2 <- nnet(V618 ~ ., data = x.tr, size = 6, decay = 0.3, maxit = 1500, MaxNWts = 4000, trace = FALSE)
a2$value
#> [1] 100.5938
b1 <- predict(a1, type = "class") # , type='raw')
mean(b1 != x.tr$V618)
#> [1] 0
b2 <- predict(a2, type = "class") # , type='raw')
mean(b2 != x.tr$V618)
#> [1] 0
b1 <- predict(a1, newdata = x.te, type = "class") # , type='raw')
mean(b1 != x.te$V618)
#> [1] 0.01909308
b2 <- predict(a2, newdata = x.te, type = "class") # , type='raw')
mean(b2 != x.te$V618)
#> [1] 0.01193317
```
Note that in this case the NN with a better objective
function (100.5937916 versus 102.1805383) achieves a better performance on 
the test set (0.012 
versus 0.019), although the
difference is rather small. Conclusions based on a 
proper CV study would be much more reliable.

You are strongly encouraged to study what happens with other
combinations of decay, number of weights and number of units
on the hidden layer, using a proper CV setting to evaluate
the results. 

<!-- #### Additional resources for discussion (refer to the lecture for context) -->

<!-- * [https://arxiv.org/abs/1412.6572](https://arxiv.org/abs/1412.6572) -->
<!-- * [https://arxiv.org/abs/1312.6199](https://arxiv.org/abs/1312.6199) -->
<!-- * [https://www.axios.com/ai-pioneer-advocates-starting-over-2485537027.html](https://www.axios.com/ai-pioneer-advocates-starting-over-2485537027.html) -->
<!-- * [https://medium.com/intuitionmachine/the-deeply-suspicious-nature-of-backpropagation-9bed5e2b085e](https://medium.com/intuitionmachine/the-deeply-suspicious-nature-of-backpropagation-9bed5e2b085e) -->


