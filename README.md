# BiNLOP

BiNLOP is a novel activation function for deep learning and machine learning use cases. 

BiNLOP is denoted as:

c = gx+(1-g)*max(-k,min(k,x)
Where g is a trainable parameter, as with k.

Benchmarks:

On a 1M parameter Transformer trained on TinyShakespeare for 7 epochs...
LOSS:
GeLU: 2.29 
BiNLOP: 2.36
Swish: 2.32

THROUGHPUT:

BiNLOP: 182K tok/s (not PyTorch native)
GeLU: 150K tok/s
Swish: 190K tok/s

