# (APPENDIX) Appendix {.unnumbered}

# PCA and alternating regression {#alt-pca}

Let $X_1, \ldots, X_n \in \mathbb{R}^p$ be the observations for which we
want to compute the corresponding PCA. Without loss of generality we can
always assume that 
$$
\frac{1}{n} \sum_{i=1}^n X_i \ = (0,\ldots,0)^\top \, ,
$$ 
so that the sample covariance matrix $S_n$ is 
$$
S_n \ = \ \frac{1}{n-1} \, \sum_{i=1}^n X_i \, X_i^\top \, .
$$ 
We saw in class that if $B \in \mathbb{R}^{p \times k}$ has in its
columns the eigenvectors of $S_n$ associated with its $k$ largest
eigenvalues, then 
$$
\frac{1}{n} \, \sum_{i=1}^n \left\| X_i - P( L_{B}, X_i
) \right\|^2 \ \le \ 
\frac{1}{n} \, \sum_{i=1}^n \left\| X_i - P( L, X_i
) \right\|^2 \, ,
$$ 
for any $k$-dimensional linear subspace $L \subset \mathbb{R}^p$
where $P( L, X)$ denotes the orthogonal projection of $X$ onto the
subspace $L$, $P( L_{B}, X) = {B} {B}^\top X$ (whenever ${B}$ is chosen
so that ${B}^\top {B} = I$) and $L_{B}$ denotes the subspace spanned by
the columns of $B$.

We will show now that, instead of finding the spectral decomposition of
$S_n$, principal components can also be computed via a sequence of
"alternating least squares" problems. To fix ideas we will consider the
case $k=1$, but the method is trivially extended to arbitrary values of
$k$.

When $k=1$ we need to solve the following problem
\begin{equation}
\min_{\left\| a \right\|=1, v \in \mathbb{R}^n} \
 \sum_{i=1}^n \left\| X_i - a \, v_i \right\|^2, 
 (\#eq:pca1)
\end{equation}
where $v = (v_1, \ldots v_n)^\top$ (in general, for any $k$ we have 
$$
\min_{ A^\top A = I, v_1, \ldots, v_n \in \mathbb{R}^k} \
 \sum_{i=1}^n \left\| X_i - A \, v_i \right\|^2 \, ).
$$ 
The objective function in Equation \@ref(eq:pca1) can also be written
as
\begin{equation}
 \sum_{i=1}^n \sum_{j=1}^p \left( X_{i,j} - a_j \, v_i \right)^2 \, , (\#eq:pca2)
\end{equation}
and hence, for a given vector $a$, the minimizing values of
$v_1, \ldots, v_n$ in Equation \@ref(eq:pca2) can be found solving $n$
separate least squares problems: 
$$
v_\ell \, = \, \arg \,  \min_{d \in \mathbb{R}}
\sum_{j=1}^p \left( X_{\ell,j} - a_j \, d \right)^2 \, , \qquad
\ell = 1, \ldots, n \, .
$$ 
Similarly, for a given set $v_1, \ldots, v_n$ the entries of $a$ can
be found solving $p$ separate least squares problems: $$
a_r \, = \, \arg \,  \min_{d \in \mathbb{R}}
\sum_{i=1}^n \left( X_{i, r} - d \, v_i \right)^2 \, , \qquad
r = 1, \ldots, p \, .
$$ We can then set $a \leftarrow a / \| a \|$ and iterate to find new
$v$'s, then a new $a$, etc.
