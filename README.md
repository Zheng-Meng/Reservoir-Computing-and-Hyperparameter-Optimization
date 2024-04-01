
<h1 align="center">Reservoir Computing and Hyperparameters Optimization</h1>

<p align="center">
<img src='images/rc.png' width='500'>
</p>

Reservoir computing (RC) is a type of recurrent neural networks (RNNs). A reservoir computer is composed of three layers: an input layer, a hidden recurrent layer, and an output layer. The distinguishing feature of RC is that only the readout weights ($W_{out}$) are trained using a linear regression, while the input weigths ($W_{in}$) and recurrent connection weights within the reservoir ($\mathcal{A}$) are determined before training. This approach gives RC a notable advantage over other RNNs, particularly in terms of rapid learning.

Now suppose we have the training data: input $u$ and the corresponding output $v$, let's first update the reservoir states $r$. Activated by the sequence of input signals $[u(1), u(2), \cdots, u(t)]$, the hidden layer state is updated step by step at the input signal time interval according to 

$r(t+1) = (1-\alpha) \cdot r(t) + \alpha \cdot \tanh[A \cdot r(t) + W_{in} \cdot u(t)]$,

where $\alpha$ is the leakage parameter and the activation function is the hyperbolic tangent function. Note that $\mathcal{A}$ is the adjacent matrix of size $n \times n$, which determines the link relationship of the reservoir network, while $r$ is a vector of size $n \times 1$, which represents the state of each node in the reservoir network. In addition, reservoir network $\mathcal{A}$ and input matrix $W_{in}$ are randomly generated before training by Gaussian distribution and uniform distribution, respectively.













