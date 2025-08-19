# BiNLOP

BiNLOP (Bipolar Nonlinear Operator) is a novel activation function for deep learning and machine learning use cases. 

BiNLOP excels at provable, rigorous and robust training with verifiable, guaranteed rewards. However, a major drawback is its computational cost. The FLOPs are calculated to be immense compared to the typical activation function like GeLU, ReLU, Swish or Mish. A future repository called BiNLOP-Light will be released to make it computationally feasible for training. 

BiNLOP is denoted as:

y = x - (1-γ) * sign(x) * relu(|x|-k)
