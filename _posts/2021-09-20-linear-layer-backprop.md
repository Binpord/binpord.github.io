# The simplest way to get formulas for backpropagation through linear layer and a confusing detail about chain rule in matrix differentiation

Today I was solving a very popular task of implementing a neural network from scratch. As part of it, I was supposed to implement forward and backward pass of a simple linear (aka fully-connected) layer.

The forward pass was easy. Suppose I have input tensor $X$ of shape $[B, N]$, where $B$ is batch size and $N$ is number of input features and I want to obtain output tensor $Y$ of shape $[B, M]$, where $M$ is the number of output features. For this purpose I allocate weight tensor $W$ of shape $[N, M]$ and bias tensor $b$ of shape $[1, M]$ and my forward pass becomes as simple as $Y = XW + b$.

Now let's consider the backward pass. As input we get tensor $\frac{\partial L}{\partial Y}$, where $L$ is the scalar loss and hence the said tensor has shape $[B, M]$ following the size of $Y$ tensor. What I need is to find following three tensors: $\frac{\partial L}{\partial X}$ so that backpropagation can continue through my layer and a pair of $\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$ so that I can update my layer's parameters.

Let's start of with $\frac{\partial L}{\partial X}$. My initial idea was to simply follow the chain rule:

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial X} = \frac{\partial L}{\partial Y} W^T
$$

And this in fact yields correct result!

---

**Quck note**. If you're not as fluent with matrix differentiation or don't want to take my word for it, you can check out this [University of Michigan's course page](https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html) for a more detailed proof.

---

So I got the correct result using the straight-forward approach! I was ecstatic! But soon my hopes were crushed. Let's consider the next derivative:
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial W} = \frac{\partial L}{\partial Y} X
$$

I did exactly the same thing I did for the previous derivative so nothing can go wrong, right? Let's consider these tensors' shapes. The tensor $\frac{\partial L}{\partial Y}$ has a shape of $[B, M]$ and tensor $X$ has a shape of $[B, N]$. Wait a minute! This means that I cannot take that dot product! Is that it? Must I fall back to element-wise differentiation and guessing the appropriate tensor operations?

Those tensors' shapes along with the fact that tensor $\frac{\partial L}{\partial W}$ should have a shape of $[N, M]$, suggest that the product should be rearranged in the following fashion:
$$
\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}
$$

And this is in fact the right answer (yet again check out the [aforementioned page](https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html) if you need a more detailed proof). But what happened to the chain rule? How come we need to rearrange derivatives and even transpose one of them?

It took quite an effort to find the answer, but I found it [in this answer](https://math.stackexchange.com/a/3850121/398144) on [Mathematics Stack Exchange](https://math.stackexchange.com/)! Here I'd like to share it with you.

First, consider the following detail: if I have a scalar function of a tensor (e.g. loss $L$ of my layer's output $Y$), then its differential $dL$ can be written as:
$$
dL = \frac{\partial L}{\partial Y} : dY
$$

where "$:$" denotes the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product) operation. In our case it means:
$$
dL = \text{Tr}\left[\frac{\partial L}{\partial Y}^T dY\right]
$$

This is not some sophisticated theorem: if you were to consider the structure of those tensors, you'd find out that:
$$
\frac{\partial L}{\partial Y} : dY = \sum_{i,j} \frac{\partial L}{\partial Y_{ij}} dY_{ij} = dL
$$

Now let's consider the differential of $Y$:
$$
Y = XW + b \Rightarrow dY = dX W + W dX + db
$$

---

For more detail please check out [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).

---

This in turn means that $dL$ can be rewritten as:
$$
dL = \frac{\partial L}{\partial Y} : dY = \frac{\partial L}{\partial Y} : (dX W + X dW + db) =
$$
$$
= \frac{\partial L}{\partial Y} : dX W + \frac{\partial L}{\partial Y} : X dW + \frac{\partial L}{\partial Y} : db
$$

If now you were to remember, what the Frobenius inner product is and the fact that matrix trace is invariant under cyclic permutations (you can find more on this [here](https://en.wikipedia.org/wiki/Trace_(linear_algebra))), you'd find out that:
$$
dL = \frac{\partial L}{\partial Y} W^T : dX + X^T \frac{\partial L}{\partial Y} : dW + \frac{\partial L}{\partial Y} : db
$$

And hence you would get the following formulas:
$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T
$$
$$
\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}
$$
$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial Y}
$$

---

**Quick note**. The most attentive probably already saw that $\frac{\partial L}{\partial b}$ in this notation has a shape of $\frac{\partial L}{\partial Y}$ which is $[B, M]$ and is way too much for our tensor $b$ of shape $[1, M]$. This happens because in reality we expanded our tensor $b$ in the sum $XW + b$ to the shape of $XW$, which is $[B, M]$. This means that you simply need to take mean value over the 0th axis in order to get gradient for you bias term.

---

That's it! This is the secret behind getting those fancy formulas for you backpropagation without "breaking" the chain rule or messing with the element-wise differentiating.