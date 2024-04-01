
<h1 align="center">Reservoir Computing and Hyperparameters Optimization</h1>

<p align="center">
<img src='images/rc.png' width='500'>
</p>

<h3>Reservoir Computing</h3>

Reservoir computing (RC) is a type of recurrent neural networks (RNNs). A reservoir computer is composed of three layers: an input layer, a hidden recurrent layer, and an output layer. The distinguishing feature of RC is that only the readout weights ($W_{out}$) are trained using a linear regression, while the input weigths ($W_{in}$) and recurrent connection weights within the reservoir ($\mathcal{A}$) are determined before training. This approach gives RC a notable advantage over other RNNs, particularly in terms of rapid learning.

Now suppose we have the training data: input $u$ and the corresponding output $v$, let's first update the reservoir states $r$. Activated by the sequence of input signals $[u(1), u(2), \cdots, u(t)]$, the hidden layer state is updated step by step at the input signal time interval according to 

$$r(t+1) = (1-\alpha) \cdot r(t) + \alpha \cdot \tanh[A \cdot r(t) + W_{in} \cdot u(t)],$$
where $\alpha$ is the leakage parameter and the activation function is the hyperbolic tangent function. Note that $\mathcal{A}$ is the adjacent matrix of size $n \times n$, which determines the link relationship of the reservoir network, while $r$ is a vector of size $n \times 1$, which represents the state of each node in the reservoir network. In addition, reservoir network $\mathcal{A}$ and input matrix $W_{in}$ are randomly generated before training by Gaussian distribution and uniform distribution, respectively.

We calculate, update, and concatenate the vector $r(t)$ characterizing the dynamical state of the reservoir network into a matrix $\mathcal{R}$. The output signal is concatenated into a matrix $V$. Then the output matrix is determined by using Tikhonov regularization as 

$$W_{out} = V \cdot R'^T (R' \cdot R'^T + \beta I)^{-1},$$

where $I$ is the identity matrix, $\beta$ is the regularization coefficient, and $R'$ is the transport of $R$.

Once we finish the training as previously describe, we can continue to get the input in the testing phase and update the reservoir state $r(t)$, calculate the output (prediction) step by step by

$$V(t) = W_{out}\cdot r(t).$$

There are various hyperparameters to be optimized (I will introduce the hyperparameters optimization later), for example, the spectral radius $\rho$ and the link probability $p$ to determine the reservoir network, the scaling factor of the input weights of the input matrix $\gamma$, the leakage parameter $\alpha$, the regularization coefficient $\beta$, and the noise level $\sigma$ injected to the training data.

<h3>Short- and Long-term Chaotic Systems Prediction by Reservoir Computing</h3>

We take two examples to get more familiar with reservoir computing: the Lorenz system and Mackey-Glass systems. 

The classic Lorenz chaotic system is a simplified model for atmospheric convection, which can be discribed by the three dimensional nonlinear differential equations:

$$\frac{dx}{dt}=\sigma_l (y-x), $$

$$\frac{dy}{dt}=x(\rho_l - z) - y,$$

$$\frac{dz}{dt}=xy-\beta_l z,$$

where $x, y,$ and $z$ are proportional to the rate of convection, the horizontal and vertical temperature variation, respectively, and $\sigma_l, \rho_l,$ and $\beta_l$ are Lorenz system parameters. We use forth-order Runge-Kutta method with $dt=0.01$ and choose the parameter values as $\sigma_l=10, \rho_l=8/3$ and $\beta=26$ to generate chaotic trajectories. Run 'reservoir.m' to train the reservoir computer and make predictions, we obtain results:







































