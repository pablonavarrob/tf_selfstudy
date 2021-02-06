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

Another way to improve the baseline model is by *dropout*. That is, dropping random values in the dense layer. *Dropout* is a very well known form of regularization (reducing the complexity) 
