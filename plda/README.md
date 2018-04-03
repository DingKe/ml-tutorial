PLDA 是一个概率生成模型，最初是为解决人脸识别和验证问题而被提出[3,5]，之后也被广泛应用到声纹识别等模式识别任务中。不同学者从不同的动机出发，提出了不尽相同的 PLDA 算法，文献[2] 在统一的框架下比较了三种 PLDA 算法变种（standard[3,6], simplified[4], two-covariance[5,8]），并在说话从识别任务上比较了它们的性能差异。

本文讨论的 PLDA 主要是基于文献 [5] 中提出的 PLDA（即 two-covariance PLDA），这也是 Kaldi 中采用的算法。

本文 1 节简单介绍 LDA，只对 PLDA 算法感兴趣的读者可以跳过。

## 1.2 降维 LDA

基本的 LDA 可能并不吸引人。LDA 得到广泛应用的一个重要原因是，它可以用来对数据进行降维。

![](http://uc-r.github.io/public/images/analytics/discriminant_analysis/LDA.jpg)
**图2. LDA 投影 【[src](http://uc-r.github.io/public/images/analytics/discriminant_analysis/LDA.jpg)】**

以两分类为例（图2），如果将数据沿着决策线（超平面）的方向投影，投影（降维）后不影响数据的分类结果，因为，非投影方向的各类的均值和（协）方差相同，并不能为分类提供有效信息。

一般地，对于 $K$ 个类别的分类问题，对于分类为效的信息集中在 $K-1$ 维的子空间上。即基于 LDA 的假设，数据可以从原来的 $d$ 维降为 $K-1$ 维（假设$K-1 <= d$）。投影方向的计算可以参见[1]第4章。

## 1.3 类内方差与类间方差

对于 LDA 降维还有另一种解释来自 Fisher：**对于原数据 X，寻找一个线性组合 $Z = w^T X$（低维投影），使得 $Z$ 的类间方差与类内方差的比值最大。**

显然，这样的投影方式使得数据在低维空间最容易区分。为此，我们需要优化瑞利熵（Rayleigh Quotient）[1]。

$$
w = \underset{w}{\mathrm{argmax}} \frac{w^T S_b w}{w^T S_w w}
$$
等价的
$$
w = \underset{w}{\mathrm{argmax}} w^T S_b w
$$
$$
s.t. w^T S_w w = 1
$$
这是一个广义特征值问题。具体地，上式可以用拉格朗日乘数求解，即寻找下式的极值：
$$
w^T S_b w - \lambda w^T S_w w
$$
求导后得到：
$$
S_b\mathbf{w} = \lambda S_w\mathbf{w}
$$

如果 $S_w$ 可逆，则有：
$$
S_w^{-1} S_b\mathbf{w} = \lambda \mathbf{w}
$$
即我们只需求解矩阵 $S_w^{-1} S_b$ 的特征向量，保留特征值最大前 $k$（不一定是 $K - 1$）个特征向量（列向量）即可以得到 $W$（$d$ 行，$k$ 列）。 

类内-类间方差的视角更加通过，首先，它并不需要假设数据分布服从高斯且协方差相同；其次，降维的维度不再依赖类别数目。

# 2. PLDA

同 LDA 一样，各种 PLDA 都将数据之间的差异分解为类内差异和类间差异，但是从概率的角度进行建模。这里，我们按照 [5] 的思路，介绍所谓 two-covariance PLDA。

假设数据 $\mathbf{x}$ 满足如下概率关系：

$$
p(\mathbf{x}|\mathbf{y}) = \mathcal{N}(\mathbf{x}|\mathbf{y}, \Phi_w)
$$

$$
p(\mathbf{y}) = \mathcal{N}(\mathbf{y}|\mathbf{m}, \Phi_b)
$$

> LDA 假设各类中心服从离散分布，离散中心的个数固定，是训练数据中已知的类别数；PLDA 假设各类中心服从一个连续分布（高斯分布）。因此，PLDA 能够扩展到未知类别，从而用于未知类别的识别与认证。

这里要求协方差矩阵 $\Phi_w$ 是**正定**的对称方阵，反映了类间（within-class）的差异； $\Phi_b$ 是**半正定** 的对称方阵，反映了类间（between-class）的差异。因此，PLDA 建模了数据生成的过程，并且同时 LDA 一样，显式地考虑了类内和类间方差。

为了推断时方便，下面推导 PLDA 的一种等价表示。
根据线性代数的基础知识， $\Phi_w$ 和  $\Phi_b$ 可以同时**合同对角化**（simultaneous  diagonalization by congruence），即存在可逆矩阵 $V$，使得 $V^T\Phi_bV=\Psi$（$\Psi$ 为对角阵）且 $V^T\Phi_wV = I$（$I$ 是单位矩阵）。对角化方法见第 4 节。

基于上述说明，PLDA 的等价表述为：
$$
\mathbf{x} = \mathbf{m} + A\mathbf{u}
$$
其中，
$$\mathbf{u} \sim \mathcal{N}(\cdot|\mathbf{v}, I)$$
$$\mathbf{v} \sim \mathcal{N}(\cdot|0, \Psi)$$

$$
A = V^{-1}
$$

$\mathbf{u}$ 是数据空间在投影空间的对应投影点。$\Psi$ 反映了类间（within-class）的差异； $I$ 反映了类间（between-class）的差异，这里被规一化。

# 3. 基于 PLDA 的推断
对于每一个观测数据 $\mathbf{x}$ 我们都可以计算对应的 $\mathbf{u} = A^{-1}(\mathbf{x} - \mathbf{m})$。 PLDA 的推断都在投影空间中进行。

给定观一组同类的测数据 $\mathbf{u}_{1,\dots,n}$，$\mathbf{v}$ 的后验概率分布为（参见 4.2.1）：
$$
p(\mathbf{v}|\mathbf{u}_{1,\dots,n}) = \mathcal{N}(\mathbf{v}|\frac{n\Psi}{n\Psi + I} \mathbf{\bar u}, \frac{\Psi}{n\Psi + I})
$$
其中，$\mathbf{\bar u} = \frac{1}{n}\sum_{i=1}^n\mathbf{u}_i$。

因此，对于未知数据点 $\mathbf{u}^p$ 以及某类的若干数据点 $\mathbf{u}^g_{1,\dots,n}$（i.i.d.），$\mathbf{u}^p$ 属于该类的似然值：
$$
p(\mathbf{u}^p|\mathbf{u}^g_{1,\dots,n}) = \mathcal{N}(\frac{n\Psi}{n\Psi + I} \mathbf{\bar u}^g, \frac{\Psi}{n\Psi + I} + I)
$$

$$
\ln p(\mathbf{u}^p|\mathbf{u}^g_{1,\dots,n}) = C - \frac{1}{2} (\mathbf{u}^p - \frac{n\Psi}{n\Psi + I} \mathbf{\bar u}^g)^T (\frac{\Psi}{n\Psi + I} + I)^{-1}(\mathbf{u}^p - \frac{n\Psi}{n\Psi + I} \mathbf{\bar u}^g)  -\frac{1}{2}\ln |\frac{\Psi}{n\Psi + I} + I|
$$

其中，$C = -\frac{1}{2}d\ln 2\pi$ 是与数据无关的常量，$d$ 是数据的维度。特殊的，$\mathbf{u}^p$ 不属于任何已知类的概率为：
$$
p(\mathbf{u}^p|\emptyset) = \mathcal{N}(\mathbf{u}^p|0, \Psi + I)
$$

$$
\ln p(\mathbf{u}^p|\emptyset) = C - \frac{1}{2} \mathbf{u}^{pT} (\Psi + I)^{-1} \mathbf{u}^p -\frac{1}{2}\ln |\Psi + I|
$$

由于 $\Psi$ 是对角阵，因此上式中各个协方差也都是对角阵，因此，似然和对数似然都很容易求得。

利用 PLDA 进行识别（recognition）方法如下：：
$$
i =  \underset{i}{\mathrm{argmax}}\ \ln p(\mathbf{u}^p|\mathbf{u}^{g_i}_{1,\dots,n}) 
$$


对于认证问题（verification），可以计算其似然比：
$$
R = \frac{p(\mathbf{u}^p|\mathbf{u}^g_{1,\dots,n})}{p(\mathbf{u}^p|\emptyset)}
$$

或似然比对数（log likelihood ratio）：

$$
\ln R = \ln p(\mathbf{u}^p|\mathbf{u}^g_{1,\dots,n}) - \ln p(\mathbf{u}^p|\emptyset)
$$
适当的选定阈值 $T$，当 $R > T$ 判定 $\mathbf{u}$ 与已知数据属于同一个类别，反之则不是。


> 这里介绍的 two-covariance PLDA 并学习一个低维空间投影[2]，这一点不同 PLDA 及 standard PLDA 和 simplified PLDA。作为近似手段，可以在投影空间中，丢弃 $\Psi$ 中对角元素较小的若干维度对应的值。

# 4. PLDA 参数估计

PLDA 中，我们需要估计的参数包括 $A$、$\Psi$ 和 $\mathbf{m}$。

## 4.1 直接求解
对于 $K$ 类共 $N$ 个训练数据 $(x_1,\dots,x_N)$。如果每类样本数量相等，都为 $n_k = N / K$，则参数有解析的估计形式 [5]。

1. 计算类内 (scatter matrix) $S_w = \frac{1}{N} \sum_k\sum_{i \in \mathcal{C}_k} (\mathbf{x}_i - \mathbf{m}_k)(\mathbf{x}_i - \mathbf{m}_k)^T$ 和 类间 $S_b = \frac{1}{N}\sum_kn_k(\mathbf{m}_k - \mathbf{m})(\mathbf{m}_k - \mathbf{m})^T$。其中，$m_k = \frac{1}{n_k} \sum_{i \in \mathcal{C}_k} \mathbf{x}_i$ 为第 $k$ 类样本均值，$m = \frac{1}{N} \sum_k\sum_{i \in \mathcal{C}_k} \mathbf{x}_i$ 为全部样本均值。
2. 计算 $S_w^{-1}S_b$ 的特征向量 $w_{1,\dots,d}$，每个特征向量为一列，组成矩阵 $W$。计算对角阵 $\Lambda_b = W^TS_bW$，$\Lambda_w = W^TS_wW$。
3. 计算 $A = W^{-T} (\frac{n}{n-1}\Lambda_w)^{1/2}$，$\Psi = \max(0, \frac{n-1}{n}(\Lambda_b/\Lambda_w) - \frac{1}{n})$。
4. 如果将维度从原数据的 $d$ 维低到 $d^\prime$ 维，则保留 $\Psi$ 的前 $d^\prime$ 大的对角元素，将其余置为零。在进行推断时，仅使用 $\mathbf{u} = A^{-1}(\mathbf{x}-\mathbf{m})$ 非零对角无素对应的 $d^\prime$ 个元素。

如果各类数据的数量不一致，则上述算法只能求得近似的参数估计。

## 4.2 期望最大化方法（EM）

这里先列出算法流程，具体推导见下文：

输入：$K$ 类 $d$ 维数据，第 $k$ 个类别包含 $n_k$ 个样本，记 $x_{ki} \in R^{d}, 1 \le k \le K$ 为 第 $k$ 个类别的第 $i$ 个样本点。

输出：$d \times d$ 对称矩阵 $\Phi_w$，$d \times d$ 对称矩阵 $\Phi_b$，$d$ 维向量 $m$。

1. 计算统计量，$N = \sum_{k=1}^K n_k$， $f_k = \sum_{i=1}^{n_k} x_{ki}$，$m = \frac{1}{N}\sum_k f_{k}$，$S = \sum_k \sum_i x_{ki}x_{ki}^T$
2. 随机初始化 $\Phi_w$，$\Phi_b$，$m$
3. 重复如下步骤至到满足终止条件：
    * 3.1 对每一个类别，计算： $\hat \Phi = (n\Phi^{-1}_w + \Phi_b^{-1})^{-1}$, $y = \hat \Phi (\Phi_b^{-1} m + \Phi_w^{-1} f)$，$yyt = \hat \Phi + y y^T$
    * 3.2 聚合计算结果：$R = \sum_k n_k \cdot yyt_k$，$T = \sum_k y_k f_k^T$，$P = \sum_k\hat \Phi_k$，$E =\sum_k (y_k - m)(y_k - m)^T$
    * 3.3 更新：$\Phi_w = \frac{1}{N} (S + R - (T + T^T))$，$\Phi_b =  \frac{1}{K}(P + E)$

> 上述算法基本与[2]所列的 two-covariance 算法相同，不过这里我们直接使用全局均值做为 $m$ 的估计。如果各类别的样本量相同，则两种方法是等价，实现比较见[代码](./plda.py)。具体公式的差异见下文。

基于 $\Phi_w$ 和 $\Phi_b$ 可以计算出 $\Psi$ 和 $A^{-1}$。对角化的方法在 4.1 节算法的2、3步已经给出[5]。

首先，计算 $\Phi_w^{-1}\Phi_b$ 的特征向量 $w_{1,\dots,d}$，每个特征向量为一列，组成矩阵 $W$。则有：
$$
\Lambda_b = W^T \Phi_b W
$$
$$
\Lambda_w = W^T \Phi_w W
$$

显然
$$
I = \Lambda_w^{-1/2}\Lambda_w \Lambda_w^{-1/2} = \Lambda_w^{-1/2} W^T \Phi_w W \Lambda_w^{-1/2}
$$

则：
$$
\Psi = \Lambda_b \Lambda_w^{-1}
$$
$$
V = W \Lambda_w^{-1/2} 
$$
$$
A^{-1} = V^T = \Lambda_w^{-1/2} W^T
$$

这里我们首先通过 EM 算法估计 $m$、$\Phi_b$ 和 $\Phi_w$。在基础上，我们可以根据上面的公式计算 $A^{-1}$ 及 $\Psi$，从而完成 PLDA 模型的训练。

Kaldi 中基于期望最大化方法（EM）[实现](https://github.com/kaldi-asr/kaldi/blob/master/src/ivector/plda.cc)了 PLDA 的参数($\Phi_b$ 和 $\Phi_w$)估计。
文献 [2] （算法2 及附录 B） 给出了估计 $\Phi_w$ 和 $\Phi_b$ 的算法流程，并给出实现[代码](https://sites.google.com/site/fastplda/ )。

### 4.2.1 隐变量 $y$ 的后验分布

回顾 PLDA 的假设：
$$
p(\mathbf{x}|\mathbf{y}) = \mathcal{N}(\mathbf{x}|\mathbf{y}, \Phi_w)
$$

$$
p(\mathbf{y}) = \mathcal{N}(\mathbf{y}|\mathbf{m}, \Phi_b)
$$


给定某类的 $n$ 个数据 $x_{1,\dots,n}$，则 $y$ 的后验分布可以为：
$$
p(y|x_{1,\dots,n}) = p(x_{1,\dots,n}|y)p(y) / p(x_{1,\dots,n}) \propto p(x_{1,\dots,n}|y)p(y)
$$

后验为两个高斯分布的乘积，因此也服从高斯。因此，我们只需要计算均值向量和方差矩阵，即可以确定后验分布。

$$
\ln p(y|x_{1,\dots,n})  = \ln p(x_{1,\dots,n}|y) + \ln p(y) = \sum_i \ln p(x_i|y) + \ln p(y) = C - 0.5 \sum_i y^T \Phi_w^{-1} y + \sum_i x_i^T \Phi_w^{-1} y - 0.5 y^T \Phi_b^{-1} y + m^T \Phi_b^{-1} y
$$

整理 $y$ 的二次项为 $0.5 y^T (n\Phi^{-1}_w + \Phi_b^{-1}) y$，对比高斯分布的二次项系数，后验的协方差矩阵为：
$$
\hat \Phi = (n\Phi^{-1}_w + \Phi_b^{-1})^{-1} 
$$

记均值为 $\hat m$，则高斯分布的一次项为：
$$
y^T \hat \Phi^{-1} \hat m
$$

整理上面的一次式有：
$$
y^T (\Phi_b^{-1} m + \Phi_w^{-1} \sum_i x_i)
$$
对比两式，令 $f = \sum_i x_i$
$$
\hat m = \hat \Phi (\Phi_b^{-1} m + \Phi_w^{-1} f)
$$

综上，后验分布为：
$$
p(y|x_{1,\dots,n}) \sim \mathcal{N}(\hat m, \hat \Phi)
$$


### 4.2.2 E step
根据上面的推导有：
$$
\hat \Phi = (n\Phi^{-1}_w + \Phi_b^{-1})^{-1} 
$$

$$
\hat m = \hat \Phi (\Phi_b^{-1} m + \Phi_w^{-1} f)
$$

根据后验概率，易得如下期望：
$$
\mathbb{E}[y] = \hat m =  \hat \Phi (\Phi_b^{-1} m + \Phi_w^{-1} f)
$$
$$
\mathbb{E}[yy^T] = \hat \Phi + \mathbb{E}[y] \mathbb{E}[y]^T
$$

### 4.2.3  M step for $\Phi_w$


对于参数 $\Phi_w$，EM 的最大化目标函数为：
$$
Q = \mathbb{E}_{y}[\ln p(x)] = \mathbb{E}_{y}[\sum_i \ln p(x_i)]
$$

对 $x_i$ 有：
$$
\ln p(x_i) = C - 0.5 \ln|\Phi_w| - 0.5 (x_i - y)^T \Phi_w^{-1} (x_i - y) = C - 0.5 \ln|\Phi_w| -0.5 x_i^{T} \Phi_w^{-1} x_i -0.5 y^{T} \Phi_w^{-1} y + 0.5 x_i^T \Phi_w^{-1} y + 0.5 y^T \Phi_w^{-1} x_i
$$

$$
\mathbb{E}_{y}[\ln p(x_i)] = C - 0.5 \ln|\Phi_w| -0.5 x_i^{T} \Phi_w^{-1} x_i -0.5 \mathrm{tr} (\mathbb{E}[yy^T] \Phi_w^{-1}) + 0.5 x_i^T \Phi_w^{-1} \mathbb{E}[y] + 0.5 \mathbb{E}[y]^T \Phi_w^{-1} x_i
$$

利用矩阵求导的知识[11]，并注意到 $\Phi_w$ 和 $\mathbb{E}[yy]$ 都是对称矩阵，于是有：
$$
\frac{\partial}{\partial \Phi_w} \mathbb{E}_{y}[\ln p(x)] = -0.5 n \Phi_w^{-1} + 0.5 \Phi_w^{-1} S \Phi_w^{-1} + 0.5n \Phi_w^{-1}\mathbb{E}[yy]^T\Phi_w^{-1} - 0.5\Phi_w^{-1} f \mathbb{E}[y]^T\Phi_w^{-1} - 0.5\Phi_w^{-1}  \mathbb{E}[y] f^T\Phi_w^{-1}
$$

其中，$S = \sum_i x_i x_i^T, f = \sum_i x_i$。令上式零，求得：

$$
n\Phi_w = S + n\mathbb{E}[yy^T] - (T + T^T)
$$
其中，$T^T = \mathbb{E}[y] f^T$

根据这个类别数据， $\Phi_w$ 的估计为：
$$
\Phi_w =\frac{1}{n}[S + n\mathbb{E}[yy^T] - (T + T^T)]
$$

上面的推导是基于一个类别的数据，如果我们有 $K$ 类，对依赖类别的变量加上相应的下标，则有：
$$
\sum_{k=1}^K n_k \Phi_w = \sum_{k=1}^K S_k + n_k \mathbb{E}[y_ky_k^T] - (T_k + T_k^T)
$$

根据所有类别数据的信息，$\Phi_w$ 的估计公式为：
$$
\Phi_w = \frac{1}{N} (S_g + R_g - (T_g + T_g^T))
$$
其中，
$$
S_g = \sum_{k=1}^K S_k =  \sum_{k=1}^K \sum_{j=1}^{n_k} x_{kj}x_{kj}^T 
$$
$$
T_g = \sum_{k=1}^K T_k = \sum_{k=1}^K \mathbb{E}[y_k] f_k^T
$$
$$
R_g = \sum_{k=1}^K  n_k \mathbb{E}[y_ky_k^T]
$$

### 4.2.4 M step for $\Phi_b$ 及 $m$

对于参数 $\Phi_b$ 和 $m$，EM 的最大化目标函数为：
$$
Q = \mathbb{E}_{y}[\ln p(y)] = \sum_{k=1}^K \mathbb{E}_{y_k}[\ln p(y_k)]
$$

对 $y_k$ 有：
$$
\ln p(y_k) = C - 0.5 \ln|\Phi_b| - 0.5 (y_k - m)^T \Phi_w^{-1} (y_k - m) = C - 0.5 \ln|\Phi_b| -0.5 y_k^{T} \Phi_b^{-1} y_k -0.5 m^{T} \Phi_b^{-1} m + 0.5 y_k^T \Phi_b^{-1} m + 0.5 m^T \Phi_b^{-1} y_k
$$

$$
\mathbb{E}_{y}[\ln p(x_i)] = C - 0.5 \ln|\Phi_b| -0.5 \mathrm{tr} (\mathbb{E}[yy^T] \Phi_b^{-1}) -0.5 m^{T} \Phi_b^{-1} m + 0.5 \mathbb{E}[y]^T \Phi_b^{-1} m + 0.5 m^T \Phi_b^{-1} \mathbb{E}[y]
$$


**a) $m$ 的估计**

类似于4.2.3，我们有：
$$
\frac{\partial}{\partial m} \mathbb{E}_{y}[\ln p(y)] = \sum_k -m^T \Phi_b^{-1} + \mathbb{E}[y]^T\Phi_b^{-1} 
$$

令上式为零，求得：
$$
m = \frac{1}{K} \sum_k \mathbb{E}[y_k]
$$

由于各类数量可能不均衡，可以加权[2]：
$$
m = \frac{1}{N} \sum_k n_k\mathbb{E}[y_k]
$$

此时，如果各类别数量相同，将 $\mathbb{E}[y] = \hat \Phi (\Phi_b^{-1} m + \Phi_w^{-1} f)$ 代入上式有：
$$
m = \frac{1}{N} \sum_{k=1}^K f_k = \frac{1}{N} \sum_{k=1}^K \sum_{i=1}^{n_k} x_{ki}
$$

即 $m$ 是已知数据的全局均值，不需要迭代。在 Kaldi 的实现中，不论各类数量，都直接使用全局均值做为 $m$ 的估计。

**b) $\Phi_b$ 的估计**

对 $\Phi_b$ 求导得：
$$
\frac{\partial}{\partial \Phi_b} \mathbb{E}_{y_k}[\ln p(y_k)] = -0.5 \Phi_b^{-1} + 0.5 \Phi_b^{-1} \mathbb{E}[y_k y_k^T] \Phi_b^{-1} + 0.5\Phi_b^{-1}mm^T\Phi_b^{-1} - 0.5 \Phi_b^{-1} \mathbb{E}[y_k]m^T \Phi_b^{-1} - 0.5 \Phi_b^{-1} m \mathbb{E}[y_k]^T \Phi_b^{-1}
$$

令上式为零，求得更新公式：
$$
\Phi_b =  \mathbb{E}[y_k y_k^T]  + mm^T - \mathbb{E}[y_k]m^T - m \mathbb{E}[{y_k}^T]
$$
注意到 $\mathbb{E}[yy^T] = \hat \Phi + \mathbb{E}[y] \mathbb{E}[y]^T$，则有：
$$
\Phi_b =  \frac{1}{K}  \sum_k(\hat \Phi_k +(\mathbb{E}[y_k] - m)(\mathbb{E}[y_k] - m)^T)
$$

> 如果考虑各类之间的权重，则有：
$$
\Phi_b =  \frac{1}{\sum_{k}w_k}  \sum_k w_k (\hat \Phi_k +(\mathbb{E}[{y_k}] - m)(\mathbb{E}[{y_k}] - m)^T)
$$

而如果我们使用 $m = \frac{1}{N} \sum_k n_k\mathbb{E}[y_k]$ 估计 $m$，则综合所有类别数据，并按照数据量加权，有[2]：
$$
\Phi_b = \frac{1}{N} R_g - mm^T
$$

# References
1. Hastie et al. [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/).
2. Sizov et al. [Unifying Probabilistic Linear Discriminant Analysis Variants in Biometric Authentication](http://cs.uef.fi/~sizov/pdf/unifying_PLDA_ssspr2014.pdf).
3. Prince & Elder. [Probabilistic Linear Discriminant Analysis for Inferences About Identity](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.97.6491&rep=rep1&type=pdf).
4. Kenny et al. [Bayesian Speaker Verification with Heavy Tailed Priors](http://www.crim.ca/perso/patrick.kenny/kenny_Odyssey2010.pdf).
5. Ioffe. [Probabilistic Linear Discriminant Analysis](https://ravisoji.com/assets/papers/Ioffe2006PLDA.pdf).
6. Shafey et al. [A Scalable Formulation of Probabilistic Linear Discriminant Analysis: Applied to Face Recognition](http://ieeexplore.ieee.org/document/6461886/).
7. Hastie & Tibshirani. [Discriminant Analysis by Gaussaian Mixtures](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.36.203).
8. Brummer & De Villiers et al. [The Speaker Partitioning Problem](https://pdfs.semanticscholar.org/3e49/e2d4b026e6bfe4def3586d8cd9b2a90ee7ed.pdf).
9. Jiang et al. [PLDA Modeling in I-vector and Supervector Space for Speaker Verification](https://pdfs.semanticscholar.org/bccb/205ca4069505aefd29fca5b5cdf3db02e3d4.pdf).
10. Brummer et al. [EM for Probabilistic LDA](https://ce23b256-a-62cb3a1a-s-sites.googlegroups.com/site/nikobrummer/EMforPLDA.pdf?attachauth=ANoY7crcscfaCA3IqQl-SOmO-MU41YCYXsPkXgoI3yS1ND6EewKJI62_YbtfycbClTO7y49zyO-s8d038nPwwrL0DlTjd5kQPDFDIoAWvQoWnSUNQUxauB78WqO70sbBK73GS0_LXtFFHxyysqoB70Rz70Y5ipRzDyfhqgAxdclS2t5xGHhK6pJoOKc_gIqGZNzt7uAK_Oi6fhmfGm4Vek-3AsJka5F0mQ%3D%3D&attredirects=0).
11. Petersen & Petersen. [The Matrix Cookbook](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf).