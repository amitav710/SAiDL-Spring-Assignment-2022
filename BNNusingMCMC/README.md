# BNN Using MCMC

The following 3 python notebooks show different approaches to get our accuracy on the noisy XOR problem using a **Bayesian Neural Network**.

<hr>

The first one being the standard approach, while the others are more discriminative. All of them have the same basic structure and model. It is only in the procedure of selection of weights through the Metropolis Hastings algorithm that there is a difference among them. The entire implementation of my code in is in **NumPy**. The functional model has one fully connected hidden layer and contains **4 neurons with appropriate dimensions to make it a fully connected layer**. Both, the input layer and the hidden layer have been equipped with biases. In the BNN class constructor, our `θ_0` is randomly initialized, following a normal distribution. Our dataset is randomly generated for training as well as testing and corresponding `y_train` and `y_test` values are also calculated using noisy XOR.
We use the sigmoid activation function in every layer. Using **Bayes Thm** we can obtain the following results:-
<hr>

>![image](https://user-images.githubusercontent.com/77532573/154806772-938c52e4-add9-48ba-88db-89583c115f93.png)
<hr>

>![image](https://user-images.githubusercontent.com/77532573/154806782-4800fb11-5022-4081-b17c-c18689ff1c95.png)
<hr>

>From Algorithm 1, we can see that our list of accepted θs is nothing but a distribution of our posterior, which we are trying to calculate. To calculate the posterior, we find out the prior and the likelihood. The product of which is directly proportional to our posterior. We use the MCMC approach of sampling as calculation of the evidence is very hard. We summarize the predictions of our BNN by model averaging, which is an ensemble prediction.

<br>

>>![image](https://user-images.githubusercontent.com/77532573/154806796-4aedc9b1-378b-4242-81c9-878e0a56b102.png)

<br>

As mentioned in the paper, a good default prior to take would be one with 0 mean and we can let sigma be 1. Our likelihood, which is defined by Bayes’ theorem as `P(data/θ)` can once again be captured in the form of a normal distribution, where the mean is nothing but the predicted value of the output of our functional model`(y_pred)`. Both, the prior and likelihood have been calculated in in their logarithmic terms, as if very negative values were powers of e, it leads to python giving NaN outputs due to possible divisions with 0 values (e^large negative values = 0 according to python).

<hr>

>![image](https://user-images.githubusercontent.com/77532573/154806808-049dad78-2968-4edf-bbec-6f3fe031b3d7.png)

<hr>

We also notice a link between regularization terms in point estimate networks and our prior. Its effects are clearly visible when we try to compute our acceptance ratio just by means of comparisons of likelihoods of two θs.

<hr>

>![image](https://user-images.githubusercontent.com/77532573/154806817-ee840583-ec2f-4afa-bca0-b69641eb7e1c.png)

<hr>
<br>

>It serves as a constraint to the posterior in a manner similar to regularizers.

We use a BBN where the coefficients (θ) are the stochastic variables that are to be sampled. The weights and biases of our functional model can be assumed to possess a normal distribution as shown:

<br>

>>![image](https://user-images.githubusercontent.com/77532573/154806832-d190e852-e8ae-4900-bda3-2026dc8f1441.png)

<br>


When we define our function `Q` to get `w_n+1`, we once again use a bell curve having the mean as `w_n` and its size being the same. This way, we construct a new `θ_n+1`, having the same shape as θ_n, using normal distribution of its individual components. This relation between `w_n+1` and `w_n` is an innate characteristic of MCMC methods, where a sample from a distribution probabilistically depends on previous samples.

<hr>

>![image](https://user-images.githubusercontent.com/77532573/154806836-9d75550e-1261-4a11-8681-906bcc4e8746.png)

<hr>

For our Metropolis Hastings algorithm, we played around with the ratio across the 3 notebooks. In the notebook where ratio is 0, we get a somewhat consistent accuracy of around `92%`. Here, the accepted set of θs is relatively large in size.

<hr>

>![image](https://user-images.githubusercontent.com/77532573/154806839-c760137e-7f16-4d9e-bf64-25b68f55a44b.png)

<hr>

Since there is no calculation of gradients in the BNN, we adopt the use of a more punishing ratio. This serves a similar purpose to that of the learning rate in the backpropagation algorithm, in the way that weights are accepted only if improvements are made >= our ratio.
In our other two notebooks, a far more punishing ratio of **10 to 15** has been used. This forces every new θ to be at least **e^10** times more proportional to the posterior as the last accepted θ. 

<hr>

>![image](https://user-images.githubusercontent.com/77532573/154806853-1bec62c0-85c8-41b6-84ee-4c1f865e4fa9.png)

<hr>

This sort of discriminative treatment of θs keeps our accepted set small and generates a lot of solid weights, which always give good performance. We need to keep in mind not to make the ratio too punishing, or else the significance of the randomly initialized θ_0 and other poor performing weights at the start will increase.
To ensure that only the best of the best weights are selected, we can choose the last n number of elements from our accepted set, which are of course, the most proportional to the posterior out of all the θs in the accepted set, as our accepted set can be sorted in a manner that maximizes proportionality with our posterior.

<hr>

>![image](https://user-images.githubusercontent.com/77532573/154806918-00fd1b61-197f-4d8b-bfe1-d335118b2a14.png)

<hr>


>This was used in the **96_2 notebook**, where the last/best 4 weights have been used to get an ensemble prediction. This procedure gives us a great final stable accuracy of 96.2%. To ensure our accepted set has a decent size despite our discriminative approach, we run the program for a large number of epochs(30,000). This increases our chances of fetching better performing θs. This is feasible as it is not a computationally expensive program. 

Overall, the assumption of normal distributions in all places seemed to provide satisfactory results, without any exception over multiple runs.
To make other possible improvements upon this model, we could try the use of other values of standard deviation wherever we have made the use of Normal distributions. We can also try and experiment with other functions to calculate our log-likelihood and our MCMC function `Q` to get `w_n+1` from `w_n`.
