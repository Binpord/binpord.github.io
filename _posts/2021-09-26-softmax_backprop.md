# Backpropagation through softmax layer



Have you ever wondered, how can we backpropagate the gradient through a softmax layer? If you were to google it, you would find lots of articles (such as [this one](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/), which helped me a lot), but most of them prove the formula of the softmax's derivative and then jump straight to the backpropagation of cross-entropy loss through the softmax layer. And while normalizing the networks' output before computing the classification loss is the most common use of softmax, those formulas have little to do with the actual backpropagation through the softmax layer itself, more like the backpropagation through the cross-entropy loss.

You sure can compute softmax as part of the loss to ease the computation and most modern libraries do that for you (e.g. see the [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) docs), but what if we were to use softmax in a middle of a network? Can we efficiently propagate gradient through such layer, if we only have the `forward` and `backward` methods and are only given our output's gradient on the backward pass? It turns out that we can and we can do this very efficiently! Let me show you such method!

However, first let's consider the straight-forward approach, which is to compute the jacobian matrix and multiply the output gradient by it:

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial X}
$$

Here $\frac{\partial L}{\partial X}$ and $\frac{\partial L}{\partial Y}$ are 2d tensors of shape $[B, N]$, where $B$ is a batch size and $N$ is a number of features and $\frac{\partial Y}{\partial X}$ is the softmax jacobian, which in such batched case is a 3d tensor of shape $[B, N, N]$. Multiplication denotes the batched matrix multiplication.

Let's implement this approach as a baseline. The derivative of the softmax can be calculated as:

$$
\frac{\partial \sigma(x)_i}{\partial x_j} = \sigma(x)_i (1_{i=j} - \sigma(x)_j)
$$

Which yields the formula for a batched jacobian:

$$
\frac{\partial Y_{ik}}{\partial X_{ij}} = Y_{ik} (1_{k=j} - Y_{ij})
$$

Please note that $i$ denotes the item in batch and computation is done for each item independently.

```python
import numpy as np

def softmax(input_, axis=1):
    exp_input = np.exp(input_)
    return exp_input / exp_input.sum(axis, keepdims=True)

# Let's fix random seed and dimensions.
rng = np.random.default_rng(42)
batch_size, n_features = 32, 512
```

```python
# Forward pass.
# Suppose we got some input (i.e. X).
input_ = rng.random((batch_size, n_features))

# Now we apply the softmax to get the output (i.e. Y).
output = softmax(input_, axis=1)
```

```python
# Backward pass.
# Suppose we got some output_grad (i.e. dL/dY).
output_grad = rng.random((batch_size, n_features))

# Now we need to compute the input_grad (i.e. dL/dX).
# For that we want to compute the jacobian (i.e. dY/dX).
identity = np.eye(output.shape[1], dtype=output.dtype)
jacobian = output[:, :, None] * (identity[None, :, :] - output[:, None, :])

# Now we compute the input_grad.
input_grad = (jacobian @ output_grad[:, :, None]).squeeze(2)
```

---

**Quick note**. If you want to make sure, that we did in fact compute the correct gradient, you can do so using the PyTorch's autograd with the following code.

```python
import torch

# Convert our random data into pytorch tensors.
torch_input = torch.tensor(input_, requires_grad=True)
torch_output_grad = torch.from_numpy(output_grad)

# Compute softmax (retain grad to check the intermediate gradient).
torch_output = torch.softmax(torch_input, axis=1)
torch_output.retain_grad()

# Sum the output with the output_grad coefficients.
# This will make the gradient torch_output.grad be equal to output_grad.
loss = (torch_output * torch_output_grad).sum()
loss.backward()
assert np.allclose(torch_output.grad.detach().numpy(), output_grad)

# Now that we have correct output_grad, let's check the input gradient.
assert np.allclose(torch_input.grad.detach().numpy(), input_grad)

# If this cell didn't raise any AssertionErrors, then we did everything right.
```

---

Now let's move on to the second approach, which will allow us to compute the $\frac{\partial L}{\partial X}$ without explicitely computing the jacobian. This is very important as it means that we wouldn't need to store possibly quite heavy 3d tensor in memory.

In order for this to work, let's write down the jacobian-based formula and substitute the $\frac{\partial Y}{\partial X}$:

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial X} \Rightarrow \frac{\partial L}{\partial X_{ij}} = \sum_{k=1}^N \frac{\partial L}{\partial Y_{ik}} \frac{\partial Y_{ik}}{\partial X_{ij}} = \sum_{k=1}^N \frac{\partial L}{\partial Y_{ik}} Y_{ik} (1_{k=j} - Y_{ij})
$$

Now let's look at the terms separately:

$$
\frac{\partial L}{\partial X_{ij}} = \sum_{k=1}^N \frac{\partial L}{\partial Y_{ik}} Y_{ik} 1_{k=j} - \sum_{k=1}^N \frac{\partial L}{\partial Y_{ik}} Y_{ik} Y_{ij}
$$

First one contains a sum out of which only one element isn't a zero because of the indicator and in second sum we can take the $Y_{ij}$ out of summation, because it doesn't contain $k$:

$$
\frac{\partial L}{\partial X_{ij}} = \frac{\partial L}{\partial Y_{ij}} Y_{ij} - \left(\sum_{k=1}^N \frac{\partial L}{\partial Y_{ik}} Y_{ik}\right) Y_{ij} = \left( \frac{\partial L}{\partial Y_{ij}} - \sum_{k=1}^N \frac{\partial L}{\partial Y_{ik}} Y_{ik} \right) Y_{ij}
$$

And this is in fact something we can compute.

```python
input_grad_v2 = (output_grad - (output_grad * output).sum(axis=1, keepdims=True)) * output
assert np.allclose(input_grad, input_grad_v2)
```

We get exact same result without any need of generating the identity matrix and/or materializing the jacobian in memory! Profiling the memory use in python can be tricky, so let's instead check the time requirements for these two methods:

```python
def backward(output, output_grad):
    identity = np.eye(output.shape[1], dtype=output.dtype)
    jacobian = output[:, :, None] * (identity[None, :, :] - output[:, None, :])
    return (jacobian @ output_grad[:, :, None]).squeeze(2)

def backward_v2(output, output_grad):
    return (output_grad - (output_grad * output).sum(axis=1, keepdims=True)) * output
```

```python
%timeit backward(output, output_grad)
```

    65.1 ms ± 13.2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


```python
%timeit backward_v2(output, output_grad)
```

    31.9 µs ± 44.3 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


We get 2000 times faster performance and this is just a moderate-sized input we are talking about! So the second version seems the way to go and allows us to propagate gradient through our softmax layer pretty efficiently.

Hope this was useful and thank you for your time!
