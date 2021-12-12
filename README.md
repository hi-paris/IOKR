# IOKR

Refs:
- Brouard, C., Shen, H., Dührkop, K., d'Alché-Buc, F., Böcker, S., & Rousu, J. (2016). Fast metabolite identification with input output kernel regression. Bioinformatics, 32(12), i28-i36.

## I. IOKR for Structured Prediction

***

In this section, you are going to implement and test an elegant method for structured prediction called **Input Output Kernel Regression (IOKR)**. This method belongs to the family of surrogate methods for structured prediction (cf. lectures).

**Kernel methods (quick reminder).** A kernel over a space $Z is a map $k : \mathcal{Z} \times \mathcal{Z} \rightarrow \mathbb{R}$ that can be written:

$$k(z, z') = \langle \phi(z),\, \phi(z')\rangle_{\mathcal{G}}$$

where $\mathcal{G}$ is a Hilbert space and $\phi: \mathcal{Z} \rightarrow \mathcal{G}$ a map. Hence, $k$ define a similarity between any couple of points in $\mathcal{Z}$. 

**Two examples of kernel.**

The most simple example of kernel is the linear kernel $k(z, z') = \langle z,\, z'\rangle_{\mathcal{Z}}$. So $\phi(x) = x$.

Another famous kernel is the gaussian kernel, defined, for a given $\sigma^2 >0$ by: $k(z, z') = \exp(-\frac{\|z-z\|^2}{2\sigma^2})$. Here there also exists a $\phi$ verifying the kernel definition above, but with valued in an infinite dimensional space! This is not an issue as we never need to explicitely compute any $\phi(x)$ (see below).

**Description of the method.** Consider that we are given two kernels, one over the input and another over the output space.

$$k_x(x,x') = \langle \phi(x),\, \phi(x')\rangle_{\mathcal{G}}$$

$$k_y(y,y') = \langle \psi(y),\, \psi(y')\rangle_{\mathcal{H}}$$ 

The IOKR method for structured prediction can be described in two steps:

- 1/Learning step. Learn the linear map $W : \mathcal{G} \times \mathcal{H}$ thanks to

$$\min_{W} \sum_{i=1}^n \|W\phi(x_i) - \psi(y_i)\|^2_{\mathcal{H}_y} + \lambda \|W\|^2_{F}$$

- 2/Testing step. For a given new inputs $x$ predict $\hat f(x) = \text{argmin}_{y \in \mathcal{Y}}\; \langle \psi(y),\,\hat W\phi(x) \rangle_{\mathcal{H}}$

Observe that the predictor $\hat f: \mathcal{X} \rightarrow \mathcal{Y}$ is non-linear w.r.t $x$ and also $y$.

**Algorithm.** Finally, one can derive a closed-form formula for $\hat f : \mathcal{X} \rightarrow \mathcal{Y}$ defined above. Here we do not ask you to derive this formula but we give it to you:

\begin{equation}
\hat f(x) = \text{argmax}_{y \in \mathcal{Y}} \; \alpha_x(x)^T M \alpha_y(y)
\end{equation}

with $\alpha_x(x) = \left(k_x(x, x_1), \dots, k_x(x, x_n)\right) \in \mathbb{R}^{n}$, $\alpha_y(y) = \left(k_y(y, y_1), \dots, k_y(y, y_n)\right) \in \mathbb{R}^{n}$, $K_x = \left(k_x(x_i, x_j)\right)_{i,j} \in \mathbb{R}^{n \times n}$, and $M = \left( K_x + \lambda I_{\mathbb{R}^n} \right)^{-1} \in \mathbb{R}^{n \times n}$.

![](img/IOKR.pdf)

**Remark (link with operator-valued kernel).** The IOKR approach is illustrated on the figure above. Observe that we perform a linear regression from $\mathcal{G}$ to $\mathcal{H}$, which can be both infinite dimensional spaces (called reproducing kernel Hilbert space) when $\phi$ and $\psi$ are defined with the non linear gaussian kernel. Moreover, notice also, that the approach corresponds to a vector-valued kernel regression from $\mathcal{X}$ to $\mathcal{H}$. This situation can be interpreted through the use of an operator-valued kernel (ovk). Here we use the ovk defined by $\tilde k_x(x,x') =  k_x(x,x') I_{\mathcal{H}}$, where $I_{\mathcal{H}}$ denotes the identity operator of $\mathcal{H}$.

***
