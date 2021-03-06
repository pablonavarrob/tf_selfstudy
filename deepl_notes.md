# Deep Learning with Tensorflow 
## Chapter 1: NN Foundations with Tensorflow 2.0 

On the first note, while Tensorflow has many features similar to other Deep learning codes, it is special because it has Keras as its high-level API, allows model deployment and ease of use in production, supports eager computation, based on graph computation and great community support.  

**Keras**: API for composing building blocks to create and train deep learning models. 

Artificial neural networks are a class of ML models that loosely resemble the central nervous system of mammals. Interconected "neurons" are organized in layers. Neurons in one layer pass messages into those in the next layer. Deep learning is a class of neural networks characterized b a significant number of neuron layers that are able to learn rather sophisticated models based on progressive levels of abstraction.  

These are a way to compute a function that maps inputs to corresponding outputs. The function is just a series of addition and multiplication operations. Howeer, together with non-linear activation functions and stacked into multiple layers, these can learn almost anything. 

#### Perceptron 
Simple algorith, that, given an input vector $x$ of $m$ values ($x_1, x_2, ... , x_m$), outputs either binary 1 or 0. Mathematically given by:

$$
f(x) = \begin{cases}
1 \, \, wx + b > 0 \\
0 \, \, \text{otherwise}
\end{cases}
$$

In the above, $w$ is the vector of weights and $wx$ is the dot product of inputs and weights, while $b$ are the bias terms. As a not, $wx + b$ defines a boundary hyperplane (subspace with dimension one less than ambien space) that changes position according to the values assigned to $w$ and $b$.

To create a model in *tf.keras*, there are three ways: sequential API, funcitonal API and model subclassing. There are three ways to initialize weights in layer: *random_uniform*, *random_normal* and *zeros*. 

#### Multi-layer perceptron 

Or MLP, network with multiple dense layers. Note that input and output are visible from outside, so all the layers in between are *hidden*. While a single layer corresponds to a simple linear function, MLPs are comprised of stacked linear functions. Activation functions are used to compute the output of a neuron. While perceptrons output either 1 or 0, learning cannot happen this way and smoother functions are needed.

#### Activation functions

- Sigmoid: $\sigma (x) = \frac{1}{1 + e^{-x}}$, has small output changes in the range between 0 and 1. Neurons can use sigmoid functions to compute te non-linear function $\sigma (z = wx + b)$. Sigmoid behaves similarly to perceptron but it outputs gradual changes. 
- Hyperbolic tangent: $\tanh (z) = \frac{e^{z} - e^{-z}}{e^{z} - e^{-z}}$, with output range between -1 and 1. 
- ReLU: rectified linear unit, defined as $f(x) = max(0, x)$. It is a non-linear function, that outputs 0 for negative values and grows linearly for positive. Very simple to implement. 
- ELU: exponential linear unit, defined as

$$
f(\alpha, x) = \begin{cases}
\alpha(e^{x} - 1)  \; \; if  \; \; x \leq 0 \\
x  \; \; if  \; \; x > 0
\end{cases}
 \; \; \text{for} \; \; \alpha > 0
$$

- LeakyReLU: defined as 

$$
f(\alpha, x) = \begin{cases}
\alpha x \; \;\text{if}  \; \; x \leq 0 \\
x  \; \; \text{if}  \; \; x > 0
\end{cases}
 \; \; \text{for}  \; \; \alpha > 0
$$

The two last functions allow for small updates even if the input is negative, which might be useful in certain conditions. 

#### Defining a simple model
While defining a simple model needs the general steps in machine learning of downloading, separating in train and test sets and scaling the data, neural networks need to steps. First, the model has to be defined using the Keras API and then a loss function and optimizer have to be defined. The loss functions, or objective, are what will be used to quantify the error made by the network and is what needs to be minimized. Some examples are:

- MSE: mean squared error. Average of all the mistakes made in each prediction. The squaring makes large errors even more evident. Also the square can add errors regardless of sign. 
- *binary_crossentropy*: binary logarithmic loss, used for binary label prediction. It is defined as:

$$
L (p,c) = -c \ln(p) - (1 - c)\ln(1 - p)
$$

- *categorical_crossentropy*: defines the multiclass logarithmic loss. Compares the distributions of the predictions with the true distribution. One way to think about multi-class logarithmic loss is to consider the true class represented as a one-hot vector, and the closer the model's output are to that vector, the lower the loss. If the true class is $c$ and the predictions are $y$, then the cross-entropy is:

$$
L (c, p) = - \sum_{i} c_i \ln(p_{i})
$$

Some choices for metrics are accuracy (ratio of number of correct predictions to the total number of input samples), precission (number of correct positive results divided by the number of positive results predicted by the classifier) and recall (number of correct positive results divided by the number of **all** samples that should have been identified as positive). While these are similar to objective functions, metrics are not used for the minimization but to evaluate the performance of the model. For the optimization, stochastic gradient descent is commonly used. It reduces the mistakes made by the network after each training epoch. 

*With the model compiled in Tensorflow, one can use .fit() and select the number of epochs or times the model will be exposed to the training set, and batch_sizes, number of training instances observed before the optimizer performs a weight update (normally there are many batches per epoch)*

#### Running simple TF net, establishing baseline and improving results with hidden layers
In the first example from the book, a simple network with one denselayer is used (softmax activation function). While the results are not bad for the baseline model, at 92% accuracy in the training and 91% in the test. The baseline model in this case, defines the result from the simplest approach and the goal becomes to improve this. In neural networks, one possiblity is to improve them by adding hidden layers and different activation functions. More hidden layers, allow the network to learn more complex patterns hidden in the data (at the expense of possible overtraining). 

Note: at some point, imprevement is impercetible and the loss seems to oscillate around a certain value. At this point, adding more layers won't improve the result further and it can be said that the result has converged. 

One way to improve the baseline model besides adding layers is by *dropout*. That is, dropping random values in the dense layer. *Dropout* is a very well known form of regularization (reducing the complexity) Another way, would be to change the algorithm used to the optimization. Tensorflow includes a lot, most common are:

- SGD: stochastic gradient descent, includes acceleration component. This acceleration helps SGD move in the relevant direction and dampen oscillations.
- Adam and RMSProp: add a momentum component, which is proven to increase convergence speed at the expense of more computation.

Even another way is to increase the amount of neurons per hidden layer. This increases the computation time expopnetially. However, the gains decrease more and more as the network grows in complexity, thus hitting diminishing returns. This can also reduce the performance as the model might be overfitted and will not generalize well, showing in a decreased accuracy. 

Batch size is another parameter that can be modelled. While regular gradient descent tries to minimize the cost function taking into account all values at once, stochastic gradient descent actually considers small batches at once. 

#### Regularization practices
Regularization is adopted mainly to avoid *overfitting*. While a model needs to perform well on training data, it can become excessively complex to capture all the relations there present. A complex model might need a lot of time to be executed and while performance on training data can be good, it might perform poorly on validation data. This is because learns relations that are inherent to the training set and nothing else. This is actually *overfitting*.

*As a general rule: if during the training we see that the loss increases on validation after an initial decrease, then there is a problem with model complexity and it is overfitting to the training data.*

The hyperparameter $\lambda \geq 0$ controls the importance of having a simple model by choosing:

$$
min: \{ loss(\text{Training data} \; | \; \text{Model}) \} + \lambda \cdot complexity(\text{Model})
$$

There are three types of regularization:

- L1 regularization, or LASSO: complexity of the model is expressed as the sum of the absolute values of the weights
- L2 regularization, of Ridge: expressed as the sum of the squares of the weights. 
- Elastic regularization: complexity is capture by a combination of the L1 and L2 techniques.  

To add regularization to the model, one can use:

```
   from tf.keras.regularizers import l2, activity_l2
   model.add(Dense(64, input_dim=64, W_regularizer=l2(0.01),
   activity_regularizer=activity_l2(0.01)))
```

##### Batch normalization

Enables an acceleration of the training by in some cases, halving the training epochs necessary, while offering some regularization. During training, weights change and therefore the inputs of the layers can also significantly change. Layers continuously readjust the weights to a different distribution for every batch. This may slow the model's trainig, and the idea is to make layer inputs more similar in distribution during epochs and batches. While sigmoid works well close to zero, gets stuck in values far away from it. If neurons fluctuate away from sigmoid 0, then said neuron may become unable to update its weights. One solution to this is to transform layer outputs into Gaussian distribution close to zero, thus achieving smaller variations from batch to batch. Activation input $x$ is center around zero by substracting the batch's mean $\mu$ from it. Result is divided by $\sigma + \epsilon$ (sum of batch's variance and a small number) to prevent division by zero. Then, a linear transformation $y = \lambda x + \beta$ ensures that the normalization is applied during trainig. For this, $\lambda$ and $\beta$ are also optimized during training. 

## Chapter 2 Tensorflow 1.X and 2.X

I guess this is important by will be done afterwards. 

## Chapter 3 Regression 

Regression is used anywhere that has an interest in drawing relationships between two or more things. It relates the independent variable and the dependent variable. Depending on the number of dependent variables, there might be different types of regression. Two important components are the *relationship* between the independent and dependent variables, and *strength of impact* of different independent variables on dependent variables. 

#### Simple linear regression

Mathematically, it involves finding a linear equation for the predicted value $\mathcal{Y}$ of the form:

$$
Y_{hat} = W*{T}X + b
$$

where $X = \{ x_1, x_2,..., x_n \}$ are the $n$ input variables and $W = \{w_1, w_2,...,w_n \}$ are the linear coefficients, while $b$ is the bias term. This bias allows the regression to provide an output in the absence of any input, allowing to shift the data left or right for better fit. The observed error is observed between the true and predicted values, which of a single sample $i$ is $e_{i} = Y_{i} - Y_{hat, i}$. Thus goal is to minimize the error fitting the coefficients $W$ and the bias $b$. 

To minimize, take partial derivatives wrt. the linear coefficients and the bias, set to zero and find an expression that depends on the average of both the dependent and independent variable, summed over all contributions. 

#### Multiple linear regression

In most cases, dependent variables depend upon multiple independent variables. Multiple linear regression fins a relationship between all the dependent variables $X$ and the dependent output $Y$, which satisfy the predicted $Y_{hat}$ value of:

$$
Y_{hat} = W*{T}X + b
$$

where $X = \{ x_1, x_2,..., x_n \}$ are the $n$ input variables and $W = \{w_1, w_2,...,w_n \}$ are the linear coefficients, while $b$ is the bias term. Coefficients are determine as well with LSS (least squares sum) method, thus minimizing the objective function:

$$
loss = \sum_{i} (Y_{i} - Y_{hat,i})^{2}
$$

summing over all the training samples. Now, there are $n+1$ equations to be considered.

#### Multivariate linear regression

In this case, independent variables affect more than one dependent variable. Mathematically, given as:

$$
\hat{Y}_{ij} = w_{0,j} + \sum_{k=1}^{p} w_{kj} w_{ik}
$$

where $i \in \left[1,...n\right]$ and $j \in \left[1,...,m\right]$. The term $\hat{Y}_{ij}$ corresponds to the $j^{\text{th}}$ predicted output corresponding to the $i^{\text{th}}$ and $w$ represents the regression coeffiients, $x_{ik}$ is the $k^{\text{th}}$ feature of the $i^{\text{th}}$. Now, the number of equations needed is $n \times m$. Solving them brute-force involves matrices and determines. Easier methods encompass gradient descent algorithms with the LSS.

#### TensorFlow estimators

API's for scalable, production-oriented solutions. Two types:

- Canned estimators: already pre-defined models
- Custom estimators: user-defined estimators. 

##### Feature column 

It's a module that acts as bridge mbetween input data and model. The input, trainig data for the estimators are passed as feature columns. Nine functions available:

- *categorical_column_with_identity*: category is one-hot-encoded and thus unique
- *categorical_column_with_vocabulary_file*: categorical input is a string and the categories are given in a file. Strings converted to numeric and then hot-encoded.
- *categorical_column_with_vocabulary_list*: categorical input is a string and the categories are defined in a list. Strings converted to numeric and then hot-encoded.
- *categorical_column_with_hash_bucket*: too many categories for hot-encoding, use hashing instead. 
- *crossed_column*: two columns combined as one feature, i.e geolocations.
- *numeric_column*: numeric feature, either single value or matrix.
- *indicator_column*: indirectly used with categorical columns, only when these can be hot-encoded.
- *embedding_column*: indirectly used with categorical columns, only when amount of categories is too large and cannot be hot-encoded.
- *bucketized_column*: used when instead of specific numeric value, split the data into different categories depending upon value.


#### Classification tasks and decision boundaries 
While in regression the aim is to model the distribution of the input data, in classification it is to group the data into classes or categories while defining decision boundaries to separate the classes. Classification is a subset of regression. 

#### Logistic regression 

Used to determine the probability of an event. The probability of the even is expressed using the sigmoid function:

$$
P \left( \hat{Y} = 1 \; | \; X = x \right) = \frac{1}{1 + e^{-\left(b + W^{T}x \right)}}
$$

where the goal is to estimate the weights $W = \{w_1, w_2,...,w_n \}$ and the bias $b$. In logistic regession, this is done via maximum likelihood estimation or a stochastic gradient descent. If $p$ is the total number of input data points, the loss is defined as a the cross-entropy term:

$$
loss = \sum_{i=1}^{p} Y_{i} \log(\hat{Y}_{i}) + \left(1 - Y_{i} \right) \log(1 - \hat{Y}_{i})
$$

For a multiclass classification problem, the cross-entropy loss is given by:


$$
loss = \sum_{i=1}^{p} \sum_{j=1}^{k} Y_{ij} \log(\hat{Y}_{ij})
$$

## Chapter 4 Convolutional Neural Networks 