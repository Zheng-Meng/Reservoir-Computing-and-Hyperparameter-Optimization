
<h1 align="center">Reservoir Computing and Hyperparameters Optimization</h1>

<p align="center">
<img src='images/rc.png' width='500'>
</p>

Reservoir computing (RC) is a type of recurrent neural networks (RNNs). A reservoir computer is composed of three layers: an input layer, a hidden recurrent layer, and an output layer. The distinguishing feature of RC is that only the readout weights ($W_{out}$) are trained using a linear regression, while the input weigths ($W_{in}$) and recurrent connection weights within the reservoir ($\mathcal{A}$) are determined before training. This approach gives RC a notable advantage over other RNNs, particularly in terms of rapid learning.

