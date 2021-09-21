# Confusing detail about chain rule in linear layer backpropagation

Have you ever been wondering, how come gradients for a linear layer $Y = XW$ have this weird formulas?

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T,\ \frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}
$$

I mean the first one is easy: we just apply the chain rule et voila:

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial X} = \frac{\partial L}{\partial Y} W^T
$$

However, if we were to try and repeat that same exact procedure for the second gradient, we would end up with the following:

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial W} = \frac{\partial L}{\partial Y} X
$$

Now if you were to check the shapes of these tensors, you would find out that such product doesn't exist! And if you were to check the correct formula, you'd find out that we need to rearrange derivatives and even transpose one of them!

There are lots of articles on the internet, including [this article](https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html) from EECS 442 course page, which prove correct formulas through element-wise differentiation. However, I've always found that such approach always fails to give me intuition I want to really grasp the underlying mechanism. So I thought that there must be some easier way. And I found one in [this answer](https://math.stackexchange.com/a/3850121/398144) on Mathematics Stack Exchange! However, it was very hard to find and required some additional research to understand completely. That's why I decided to write this post to share it with you.

First, consider the following detail: if you have a scalar function of a tensor (e.g. loss $L$ of the layer's output $Y$), then its differential $dL$ can be written as:

$$
dL = \frac{\partial L}{\partial Y} : dY
$$

Here "$:$" denotes the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product) operation, which in our case can be rewritten as:

$$
dL = \text{Tr}\left[\frac{\partial L}{\partial Y}^T dY\right]
$$

This is not some sophisticated theorem: if you were to consider the structure of those tensors, you'd find out that:

$$
\frac{\partial L}{\partial Y} : dY = \sum_{i,j} \frac{\partial L}{\partial Y_{ij}} dY_{ij} = dL
$$

Now let's consider the differential of $Y$:

$$
Y = XW \Rightarrow dY = dX W + W dX
$$

This means that $dL$ can be rewritten as:

$$
dL = \frac{\partial L}{\partial Y} : dY = \frac{\partial L}{\partial Y} : (dX W + X dW) =
$$

$$
= \frac{\partial L}{\partial Y} : dX W + \frac{\partial L}{\partial Y} : X dW
$$

If now you were to remember, what the Frobenius inner product is and the fact that matrix trace is invariant under cyclic permutations, you'd find out that:

$$
dL = \frac{\partial L}{\partial Y} W^T : dX + X^T \frac{\partial L}{\partial Y} : dW
$$

And that's it! This is how you get the correct formulas without "breaking" the chain rule or messing with the element-wise differentiating:

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T,\ \frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}
$$

Hope that you found this useful and thank you for your time!