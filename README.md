# Contextual-Bandits
Simulating Multiple Loggers 


## Basic Settings
Suppose we are given $n$ input vectors $x_1, \ldots, x_n$ with standardized bivariate normal features
```math
\begin{align}
x_i \sim \mathcal{N}\left([0,0], [\begin{smallmatrix}1 & 0 \\ 0 & 1 \end{smallmatrix}] \right)  && i = 1,2, \ldots n \ .
\end{align}
```

For input $i$, define an affine reward function pulling each arm (0 or 1)
```math
\delta_i \equiv \delta(x_i, y_i) = \begin{cases} \alpha_1 + x_i^T \beta_1 + \epsilon_{i,1} & \text{for } y_i = 1 \\
\alpha_0 + x_i^T \beta_0 + \epsilon_{i,0} & \text{for } y_i = 0\end{cases} \ ,
```
where $\epsilon_{i, 0}, \epsilon_{i, 1}$ are intrinsic noises.

Further define a logistic policy determined by parameter $\gamma$
```math
\pi(x) = \mathbb{P}(y = 1 \mid x)= \frac{1}{1+\text{exp}(-x^T\gamma)} \ ,
```
so that $\pi(y_i = 1 \mid x_i) \equiv \pi(x_i)$ and $\pi(y_i = 0 \mid x_i) \equiv 1 -  \pi(x_i)$.

Then, we can obtain logged data (set of tuples) according to the rules defined above
```math
D = \{(x_1, y_1, \delta_1, p_1), (x_2, y_2, \delta_2, p_2), \ldots , (x_n, y_n, \delta_n, p_n)\} \ ,
```
where $y_i \in \{0,1\}$ and $p_i \equiv \pi(y_i \mid x_i)$. 


## Evaluating Utility

From the $\pi$-logged data, we may want to estimate the utility of a new policy $\bar\pi$
```math
U(\bar\pi) = \int\int_{x,y} \mathbb{P}(x)\bar\pi(y \mid x) \delta(x,y) \ .
```

1.   Monte Carlo Expectation (SAPE (sample average policy effect
) - optimal if $\delta$ is known)
```math
\hat U_{MC}(\bar \pi) = \frac{1}{n} \sum_{i=1}^n \left(\bar\pi(y_i = 1|x_i)\delta(x_i, 1) + \bar\pi(y_i = 0|x_i)\delta(x_i, 0)\right) \ .
```
2.   Inverse Propensity Scoring
```math
\hat U_{IPS}(\bar\pi) = \frac{1}{n} \sum_{i=1}^n \delta(x_i, y_i) \frac{\bar\pi(y_i \mid x_i)}{p_i} \ .
```
3.   Direct Method
```math
\hat U_{DM}(\bar \pi) = \frac{1}{n} \sum_{i=1}^n \left(\bar\pi(y_i = 1|x_i)\hat\delta_1(x_i)+ \bar\pi(y_i = 0|x_i)\hat \delta_0(x_i)\right) \ ,
```
where $\hat\delta_1(\cdot)$, $\hat\delta_0(\cdot)$ are linear regression estimates of $\delta(x, 1)$, $\delta(x, 0)$, respectively. 

4.   Doubly Robust Estimator 
```math
\hat U_{DR}(\bar\pi) = \frac{1}{n} \sum_{i=1}^n \left[\frac{\bar\pi(y_i \mid x_i)}{p_i}\left(\delta(x_i, y_i) -\hat\delta(x_i, y_i)\right)\right] +\frac{1}{n} \sum_{i=1}^n \left(\bar\pi(y_i = 1|x_i)\hat\delta_1+ \bar\pi(y_i = 0|x_i)\hat \delta_0\right) \ .
```
5.   Kernel Optimal Matching
```math
\hat U_{KOM} (\bar \pi) = \frac{1}{n} \sum_{i=1}^n W^*_i \delta_i \ .
```
where
```math
W^*  = \underset{W \in \mathcal{W}}{\text{argmin }} 
\text{CMSE}(\hat U_{W,f}, \pi)  \ ,
```
and CMSE is the conditional mean square error of any weighted estimator from class $\mathcal{W}$. \\
We consider the simple case when $\delta_t$ has a Gaussian process prior with mean $f_t$ and covariance $\gamma_t \mathcal{k}_t$. Then, CMSE has a simple form (Kallus, 2018):
```math
\gamma_1 \mathfrak{B}_1^2(W, \pi_1, \|\cdot\|_{\mathcal{K}_1}) + \gamma_0 \mathfrak{B}_0^2(W, \pi_0, \|\cdot\|_{\mathcal{K}_0}) + \frac{1}{n^2} W^T \Sigma W
```
Here, $\gamma, \Sigma$ are hyperparameters of Mahalanobis kernel and $`\mathfrak{B}_t^2(W, \pi_t, \|\cdot\|_{\mathcal{K}_t})`$ measures the relative worst-case discrepancy between $f$-moments of $t$-treated group (i.e., arm $t$) and the whole sample in the ball of RKHS. We can further simplify the objective via matrix expression. 
```math
(I_1W - \Pi_1)^TK_1(I_1W - \Pi_1) + (I_0W - \Pi_0)^TK_0(I_0W - \Pi_0) +  \frac{1}{n^2} W^T \Sigma W
```
where $\Pi_t$ is the $n$-length vector with $\pi(y_i=t\mid x_i)$ in $i^{\text{th}}$ entry and $I_t$
is the $n\times n$ diagonal matrix with $\mathbb{I}[y_i = t]$ in the $i^{\text{th}}$ diagonal entry. Notice that the problem is a QP and indeed solvable. 


## Evaluation from Multiple Loggers
We use a similar setting as above. From the set $D = \cup_{i=1}^m D_i$ of $\pi_i$-logged data 
```math
\begin{align*}
D^i = \{(x^i_1, y^i_1, \delta^i_1, p^i_1), (x^i_2, y^i_2, \delta^i_2, p^i_2), \ldots , (x^i_{n_i}, y^i_{n_i}, \delta^i_{n_i}, p^i_{n_i})\} && i = 1, 2, \ldots, m \ ,
\end{align*}
```
we may want to estimate the utility of a new policy $\bar\pi$
```math
U(\bar\pi) = \int\int_{x,y} \mathbb{P}(x)\bar\pi(y \mid x) \delta(x,y)
```

1.   Monte Carlo Expectation (SAPE - optimal if $\delta$ is known)
```math
\hat U_{MC}(\bar \pi) = \frac{1}{n} \sum_{i=1}^m\sum_{j=1}^{n_i} \left(\bar\pi(y^i_j = 1|x^i_j)\delta(x^i_j, 1) + \bar\pi(y^i_j = 0|x^i_j)\delta(x^i_j, 0)\right)
```

### "Naive" Estimators (Same as in Single Logging Policy Above)
2.   Inverse Propensity Scoring
```math
\hat U_{IPS}(\bar\pi) = \frac{1}{n} \sum_{i=1}^m\sum_{j=1}^{n_i}  \delta(x^i_j, y^i_j) \frac{\bar\pi(y^i_j \mid x^i_j)}{p^i_j} 
```
3.   Direct Method
```math
\hat U_{DM}(\bar \pi) = \frac{1}{n} \sum_{i=1}^m\sum_{j=1}^{n_i} \left(\bar\pi(y^i_j = 1|x^i_j)\hat\delta_1(x^i_j) + \bar\pi(y^i_j = 0|x^i_j)\hat\delta_0(x^i_j)\right) \ ,
```
where $\hat\delta_1(\cdot)$, $\hat\delta_0(\cdot)$ are linear regression estimates of $\delta(x, 1)$, $\delta(x, 0)$, respectively. 

4.   Doubly Robust Estimator 
```math
\hat U_{DR}(\bar\pi) = \frac{1}{n} \sum_{i=1}^m\sum_{j=1}^{n_i}  \left[\frac{\bar\pi(y^i_j \mid x^i_j)}{p^i_j}\left(\delta(x^i_j, y^i_j) -\hat\delta(x^i_j, y^i_j)\right)\right] +\frac{1}{n} \sum_{i=1}^m\sum_{j=1}^{n_i}  \left(\bar\pi(y^i_j = 1|x^i_j)\hat\delta_1+ \bar\pi(y^i_j = 0|x^i_j)\hat \delta_0\right)
```
5.   Kernel Optimal Matching
```math
\hat U_{KOM} (\bar \pi) = \frac{1}{n} \sum_{i=1}^m\sum_{j=1}^{n_i}  W^{*i}_j\delta(x^i_j, y^i_j) \ ,
```
where
```math
W^*  = \underset{W \in \mathcal{W}}{\text{argmin }} 
\text{CMSE}(\hat U_{W,f}, \pi) 
```
and CMSE is the conditional mean square error of any weighted estimator from class $\mathcal{W}$. \\
We consider the simple case when $\delta_t$ has a Gaussian process prior with mean $f_t$ and covariance $\gamma_t \mathcal{k}_t$. Then, CMSE has a simple form (Kallus, 2018):
```math
\gamma_1 \mathfrak{B}_1^2(W, \pi_1, \|\cdot\|_{\mathcal{K}_1}) + \gamma_0 \mathfrak{B}_0^2(W, \pi_0, \|\cdot\|_{\mathcal{K}_0}) + \frac{1}{n^2} W^T \Sigma W
```
Here, $\gamma, \Sigma$ are hyperparameters of Mahalanobis kernel and $`\mathfrak{B}_t^2(W, \pi_t, \|\cdot\|_{\mathcal{K}_t})`$ measures the relative worst-case discrepancy between $f$-moments of $t$-treated group (i.e., arm $t$) and the whole sample in the ball of RKHS . We can further simplify the objective via matrix expression. 
```math
(I_1W - \Pi_1)^TK_1(I_1W - \Pi_1) + (I_0W - \Pi_0)^TK_0(I_0W - \Pi_0) +  \frac{1}{n^2} W^T \Sigma W \ ,
```
where $\Pi_t$ is the length-$n$ vector with $\pi(y_i=t\mid x_i)$ in $i^{\text{th}}$ entry and $I_t$
is the $n\times n$ diagonal matrix with $\mathbb{I}[y_i = t]$ in the $i^{\text{th}}$ diagonal entry. Notice that the problem is a QP and indeed solvable. 

### Some Alternative Estimators (Agarwal et al. 2017)
6.  Balanced IPS Estimator
```math
\hat U_{Bal}(\bar \pi) = \frac{1}{n} \sum_{i=1}^m\sum_{j=1}^{n_i}  \delta(x^i_j, y^i_j) \frac{\bar\pi(y^i_j \mid x^i_j)}{\pi_{avg}(y^i_j \mid x^i_j)} \ ,
```
where 
```math
\pi_{avg}(y \mid x) = \frac{\sum_{i=1}^{m} n_i \ \pi_i(y \mid x)}{n}`$
```

7. Weighted IPS Estimator
```math
\hat U_{Weight}(\bar \pi) = \sum_{i=1}^m\lambda^*_i\sum_{j=1}^{n_i}  \delta(x^i_j, y^i_j) \frac{\bar\pi(y^i_j \mid x^i_j)}{p^i_j} \ ,
```
where 
```math
\lambda^*_i = \frac{1}{\sigma^2_{\delta}(\bar\pi \| \pi_i) \ \sum_{j=1}^{m} \frac{n_j}{\sigma^2_{\delta}(\bar\pi \| \pi_j)}}
```
and 
$\sigma^2_{\delta}(\bar\pi \| \pi)$ is the divergence between $\bar\pi$ and $\pi$ defined by
```math
\sigma^2_{\delta}(\bar\pi \| \pi) \equiv \text{Var}_{x,y} \left[\delta(x,y)\frac{\bar\pi(y \mid x)}{\pi(y \mid x)}\right] \ .
```

## Simpler Problem (A Variant of Missing Data Problem)

Suppose we are given i.i.d. covariates $X_1, \ldots X_T$ and the corresponding partially observed i.i.d. outcomes $Y_1, \ldots Y_T$. Here, 
```math
Y_t = \begin{cases} Y_t^* &&C_t(X_t) = 1 \\ N/A && C_t(X_t) = 0\end{cases}
```
and $C_t$ depends on the historical data $X_{1:(t-1)}, Y_{1:(t-1)}$. 

### Evaluating $\mathbb{E}[Y]$

1. Naive Estimator
```math
\frac{\sum_{t=1}^T \mathbb{I}(C_t(X_t)=1)Y_t}{\sum_{t=1}^T \mathbb{I}(C_t(X_t)=1)}
```
2. Propensity Score from History
```math
\frac{1}{T} \sum_{t=1}^T \frac{\mathbb{I}(C_t(X_t)=1)Y_t}{\mathbb{P}(C_t(X_t) = 1 \mid X_{1:t})}
```
3. History-free Propensity Score
```math
\frac{1}{T} \sum_{t=1}^T \frac{\mathbb{I}(C_t(X_t)=1)Y_t}{\mathbb{P}(C_t(X_t) = 1 \mid X_{t})}
```
4. Averaging Propensity Score
```math
\frac{1}{T} \sum_{t=1}^T \frac{\mathbb{I}(C_t(X_t)=1)Y_t}{\frac{1}{T}\sum_{t'=1}^T\mathbb{P}(C_{t'}(X_t) = 1 \mid X_{t})}
```
