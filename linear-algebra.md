# Linear Algebra

## 1 Matrix Algebra

### 1.1 Basic Concepts

* $|A^T| = |A|$
* $|\lambda A| = \lambda^n |A|$
* $|AB| = |A||B|$
* $|A^{-1}| = \frac{1}{|A|}$

### 1.2 Multiplication

$$
A = \begin{bmatrix} a_{1,1} & a_{1,2} \\ a_{2,1} & a_{2,2} \end{bmatrix}
$$

$$
B = \begin{bmatrix} b_{1,1} & b_{1,2} \\ b_{2,1} & b_{2,2} \end{bmatrix}
$$

$$
C = AB = \begin{bmatrix} a_{1,1}b_{1,1} + a_{1,2}b_{2,1} & a_{1,1}b_{1,2} + a_{1,2}b_{2,2} \\ a_{2,1}b_{1,1} + a_{2,2}b_{2,1} & a_{2,1}b_{1,2} + a_{2,2}b_{2,2} \end{bmatrix}
$$

* $A(B+C) = AB + AC$
* $\lambda(AB) = (\lambda A)B = A(\lambda B)$
* $ABC = A(BC)$
* $(AB)^T = B^TA^T$

### 1.3 Inverse

* 行列式为0的矩阵不可逆

$$
AA^{-1} = A^{-1}A = E \\ |AA^{-1}|=|A||A^{-1}|=1 \\ A^{-1} = \frac{1}{|A|}A^*
$$

其中，$A^\*$为伴随矩阵:

* $A^\* = \begin{bmatrix} C\_{1,1} & C\_{2,1} \ C\_{1,2} & C\_{2,2} \end{bmatrix}$
* $C\_{i,j}$为$A$的代数余子式
* $C\_{i,j} = (-1)^{i+j}|A\_{i,j}|$
* When $A$ is a square matrix, $A$ is invertible if and only if $A$ is full rank.
* $A$ is full rank if and only if $|A| \neq 0$
* 高斯消元法求行列式
* Gauss-Jordan Elimination
* $\[A|I] \rightarrow \[I|A^{-1}]$

### 1.4 Transpose

* $|A^T| = |A|$
* $(A^T)^T = A$
* $(AB)^T = B^TA^T$
* $(A+B)^T = A^T + B^T$
* $(\lambda A)^T = \lambda A^T$
* $(A^{-1})^T = (A^T)^{-1}$

### 1.5 Block Matrix

* $A = \begin{bmatrix} A\_{11} & A\_{12} \ A\_{21} & A\_{22} \end{bmatrix}$
* $B = \begin{bmatrix} B\_{11} & B\_{12} \ B\_{21} & B\_{22} \end{bmatrix}$
* $AB = \begin{bmatrix} A\_{11}B\_{11} + A\_{12}B\_{21} & A\_{11}B\_{12} + A\_{12}B\_{22} \ A\_{21}B\_{11} + A\_{22}B\_{21} & A\_{21}B\_{12} + A\_{22}B\_{22} \end{bmatrix}$
* $A+B = \begin{bmatrix} A\_{11}+B\_{11} & A\_{12}+B\_{12} \ A\_{21}+B\_{21} & A\_{22}+B\_{22} \end{bmatrix}$

## 2 Determinant

### 2.1 Laplace Expansion

* 设$B=(b\_{i,j})$是一个$n$阶方阵

$$
|B| = \sum_{j=1}^{n}(-1)^{i+j}b_{i,j}|B_{i,j}|
$$

其中，$B\_{i,j}$是$B$去掉第$i$行第$j$列后的矩阵 例如：

$$
B = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
$$

$$
|B| = 1 \times \begin{vmatrix} 5 & 6 \\ 8 & 9 \end{vmatrix} - 2 \times \begin{vmatrix} 4 & 6 \\ 7 & 9 \end{vmatrix} + 3 \times \begin{vmatrix} 4 & 5 \\ 7 & 8 \end{vmatrix}
$$

### 2.2 solvability of algebraic equations via determinants

行列式决定了矩阵对应的线性方程组是否有解

* 如果行列式不为0，则方程组有唯一解
* 如果行列式为0，则方程组无解或有无穷多解
* Gauss Elimination

$$
\begin{bmatrix} 1 & 2 & 1 & | & 2 \\ 3 & 8 & 1 & | & 12 \\ 0 & 4 & 1 & | & 2 \end{bmatrix}
$$

$$
\begin{bmatrix} 1 & 2 & 1 & | & 2 \\ 0 & 2 & -2 & | & 6 \\ 0 & 4 & 1 & | & 2 \end{bmatrix}
$$

$$
\begin{bmatrix} 1 & 2 & 1 & | & 2 \\ 0 & 2 & -2 & | & 6 \\ 0 & 0 & 5 & | & 10 \end{bmatrix}
$$

### 2.3 Cramer's Rule

如果线性方程组

$$
\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \cdots \\ a_{n1}x_1 + a_{n2}x_2 + \cdots + a_{nn}x_n = b_n \end{cases}
$$

的系数行列式$|A| \neq 0$，则方程组有唯一解：

$$
x_i = \frac{|A_i|}{|A|}
$$

其中，$A\_i$是将$A$的第$i$列替换为$b$得到的矩阵

### 2.4 Interpretation as volume of parallelepiped

对于二阶行列式$A$,$A=\begin{bmatrix} a\_{1,1} & a\_{1,2} \ a\_{2,1} & a\_{2,2} \end{bmatrix}$，$|A|$表示以$a\_{1,1}$和$a\_{1,2}$为边的平行四边形的面积

对于三阶行列式$A$,$A=\begin{bmatrix} a\_{1,1} & a\_{1,2} & a\_{1,3} \ a\_{2,1} & a\_{2,2} & a\_{2,3} \ a\_{3,1} & a\_{3,2} & a\_{3,3} \end{bmatrix}$，$|A|$表示以$a\_{1,1}$,$a\_{1,2}$和$a\_{1,3}$为边的平行六面体的体积

## 3 Vector Space

向量空间$R^n$由所有的$n$维向量组成，向量中每个元素都是实数。

1. 如果$u$和$v$是$R^n$中的向量，$\lambda$是实数，则$u+v$和$\lambda u$也是$R^n$中的向量
2. 向量空间中的向量满足交换律和结合律
3. 向量空间中存在零向量，对于任意向量$u$，$u+0=u$
4. 对于任意向量$u$，存在$-u$，使得$u+(-u)=0$

### 3.1 Linear Independence

线性无关一般指向量的线性无关，指一组向量中任意一个向量都不能由其它几个向量线性表示，即：

对于$u, v, w$，不存在$\lambda\_1, \lambda\_2, \lambda\_3$，使得$\lambda\_1u + \lambda\_2v + \lambda\_3w = 0$

* 从矩阵上理解，线性无关是矩阵方程$Ax=0$只有零解
* 相互独立的几个向量，$v\_1, v\_2, \cdots, v\_n$，如果$\lambda\_1v\_1 + \lambda\_2v\_2 + \cdots + \lambda\_nv\_n = 0$，则$\lambda\_1 = \lambda\_2 = \cdots = \lambda\_n = 0$
* 一个矩阵是列满秩矩阵，则说明该矩阵的主元数目等于列数，即矩阵的列向量线性无关
* 如果我们有一个$m \times n$的矩阵$A$，$n>m$，列向量空间的维度最多为$m$，即最多有$m$个线性无关的列向量，那么在$m$维空间中，任意$n$个向量一定线性相关

### 3.2 basis

* 一个向量空间的基是一组向量，这一组向量必须满足如下两个特点：

1. 这组向量相互独立
2. 这组向量可以生成整个向量空间

### 3.3 invertible matrix and singular matrix

* 如果方阵的列向量能够成为一组基，化简后能够得到它的秩$r=n$，即满秩，这样的矩阵称为可逆矩阵
* 如果不能成为一组基，即$r\<n$，不满秩，这样的矩阵称为奇异矩阵

### 3.4 dimension

一个空间的基含有的向量数量是一样的，这个数量被称为维度，记作dim

### 3.5 span

* 一个向量空间的span是指这个向量空间中所有可能的线性组合 例如，$v\_1 = \begin{bmatrix} 1 \ 0 \end{bmatrix}$，$v\_2 = \begin{bmatrix} 0 \ 1 \end{bmatrix}$，则$span(v\_1, v\_2)$是所有的二维向量

## 4 Normed and Inner product space

### 4.1 Normed space

我们希望能够比较向量的大小。、 可以定义一个空间，让它同时具有向量空间和度量空间的性质，这样的空间称为范数空间。

* Normed space = Vector space + Norm
* 范数可以看作向量的长度，即对向量大小的度量方式
  * $L\_0$范数：向量中非零元素的个数
  * $L\_1$范数：向量中所有元素的绝对值之和
  * $L\_2$范数：向量中所有元素的平方和再开方
  * $L\_{\infty}$范数：向量中所有元素的绝对值的最大值
  * $L\_p$范数：向量中所有元素的绝对值的p次方和再开p次方

范数的性质：

* 非负性：$||x|| \geq 0$，且$||x|| = 0$当且仅当$x=0$
* 齐次性：$||\lambda x|| = |\lambda| ||x||$
* 三角不等式：$||x+y|| \leq ||x|| + ||y||$

### 4.2 Inner product space

内积空间 = 线性空间 + 内积

内积空间满足的条件：

* 对于任意向量$x, y, z$和任意标量$\alpha, \beta$，有：
  * $\langle x, y \rangle = \langle y, x \rangle$
  * $\langle x, \alpha y + \beta z \rangle = \alpha \langle x, y \rangle + \beta \langle x, z \rangle$
  * $\langle x, x \rangle \geq 0$，且$\langle x, x \rangle = 0$当且仅当$x=0$

### 4.3 Cauchy-Schwarz inequality

对于内积空间中的任意两个向量$x, y$，有：

$$
|\langle x, y \rangle| \leq ||x|| \cdot ||y||
$$

### 4.4 Induced norm

诱导范数是由内积定义的范数，对于内积空间中的向量$x$，有：

$$
||x|| = \sqrt{\langle x, x \rangle}
$$

### 4.5 Orthogonality

两个向量$x, y$正交，如果$\langle x, y \rangle = 0$

* $v^Tw = 0$，则$v$和$w$正交
* $||v+w||^2 = ||v||^2 + ||w||^2$，则$v$和$w$正交

### 4.6 Orthonormal basis

标准正交基

在$n$维欧式空间中，如果存在一组向量$e\_1, e\_2, \cdots, e\_n$，满足：

* $||e\_i|| = 1$
* $\langle e\_i, e\_j \rangle = 0$，$i \neq j$
* $span(e\_1, e\_2, \cdots, e\_n) = R^n$

则称$e\_1, e\_2, \cdots, e\_n$是$R^n$的标准正交基

### 4.7 Gram-Schmidt orthonormalization

对于一组线性无关的向量$u\_1, u\_2, \cdots, u\_n$，我们可以通过Gram-Schmidt正交化方法得到一组标准正交基$e\_1, e\_2, \cdots, e\_n$：

1. $e\_1 = \frac{u\_1}{||u\_1||}$
2. $e\_2 = \frac{u\_2 - \langle u\_2, e\_1 \rangle e\_1}{||u\_2 - \langle u\_2, e\_1 \rangle e\_1||}$
3. $e\_3 = \frac{u\_3 - \langle u\_3, e\_1 \rangle e\_1 - \langle u\_3, e\_2 \rangle e\_2}{||u\_3 - \langle u\_3, e\_1 \rangle e\_1 - \langle u\_3, e\_2 \rangle e\_2||}$
4. $\cdots$

例子：

$$
u_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, u_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, u_3 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}
$$

$$
e_1 = \frac{u_1}{||u_1||} = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{bmatrix}
$$

$$
e_2 = \frac{u_2 - \langle u_2, e_1 \rangle e_1}{||u_2 - \langle u_2, e_1 \rangle e_1||} = \begin{bmatrix} 1/\sqrt{6} \\ -1/\sqrt{6} \\ 2/\sqrt{6} \end{bmatrix}
$$

$$
e_3 = \frac{u_3 - \langle u_3, e_1 \rangle e_1 - \langle u_3, e_2 \rangle e_2}{||u_3 - \langle u_3, e_1 \rangle e_1 - \langle u_3, e_2 \rangle e_2||} = \begin{bmatrix} -1/\sqrt{3} \\ 1/\sqrt{3} \\ 1/\sqrt{3} \end{bmatrix}
$$

其中：

* $\langle u\_2, e\_1 \rangle = \begin{bmatrix} 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1/\sqrt{2} \ 1/\sqrt{2} \ 0 \end{bmatrix} = 1/\sqrt{2}$

## 5 Linear Maps

线性变换

### 5.1 Matrix representation of linear maps in finite-dimensional spaces

* 伸缩

$$
A = \begin{bmatrix} c & 0 \\ 0 & c \end{bmatrix}
$$

* 翻折

$$
A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

* 旋转

$$
A = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}
$$

* 投影

$$
A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}
$$

* 镜像

$$
A = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

### 5.2 kernel

假设有一个$n \times m$的矩阵$A$，$A$的kernel是指所有满足$Ax=0$的向量$x$的集合，记作$ker(A)$

### 5.3 range

$A$的range是指所有满足$Ax=b$的向量$b$的集合，记作$range(A)$

### 5.4 dimension formula

对于一个线性变换$T: V \rightarrow W$，$V$和$W$是有限维向量空间，有：

$$
dim(V) = dim(ker(T)) + dim(range(T))
$$

## 6 Eigenvalue Problem

### 6.1 Eigenvalues and eigenvectors

对于一个$n \times n$的矩阵$A$，如果存在一个标量$\lambda$和一个非零向量$x$，使得：

$$
Ax = \lambda x
$$

则称$\lambda$是$A$的特征值，$x$是$A$的特征向量

计算特征值只需要解方程$|A-\lambda I|=0$，其中$I$是单位矩阵

针对每个特征值，通过求解$(A-\lambda I)x=0$，可以得到对应的特征向量

* 该方程组的解的个数取决于矩阵$A-\lambda I$的秩
  * 如果秩为$n$，则有唯一解
  * 如果秩为$r$，特征向量的个数为$n-r$

### 6.2 Diagonalization

如果一个$n \times n$的矩阵$A$有$n$个线性无关的特征向量，那么$A$可以对角化，即：

$$
A = PDP^{-1}
$$

其中，$P$是由$A$的特征向量组成的矩阵，$D$是由$A$的特征值组成的对角矩阵

### 6.3 Spectral theorem for symmetric matrices

对于一个对称矩阵$A$，存在一个正交矩阵$P$，使得：

$$
A = PDP^T
$$

其中，$D$是$A$的特征值组成的对角矩阵

### 6.4 Similarity

如果存在一个可逆矩阵$P$，使得：

$$
B = PAP^{-1}
$$

则称$A$和$B$是相似的

### 6.5 Generalized eigenvectors

对于一个$n \times n$的矩阵$A$，如果存在一个标量$\lambda$和一个非零向量$x$，使得：

$$
(A-\lambda I)^k x = 0
$$

则称$x$是$A$的$k$阶广义特征向量

### 6.6 Jordan form

对于一个$n \times n$的矩阵$A$，如果$A$有$k$个线性无关的特征向量，但是没有$n$个线性无关的特征向量，那么$A$的Jordan标准形式是：

$$
J = \begin{bmatrix} J_1 & 0 & 0 \\ 0 & J_2 & 0 \\ 0 & 0 & J_3 \end{bmatrix}
$$

## 7 最小二乘法

找到一条直线，使得所有点到直线的距离之和最小

斜率m:

$$
m = \frac{n\Sigma_{xy} - \Sigma_x\Sigma_y}{n\Sigma_{x^2} - \Sigma_x^2}
$$

$$
b = \frac{\Sigma_y - m\Sigma_x}{n}
$$

Hessian

* 2x2

$$
H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} \end{bmatrix}
$$

* 3x3

$$
H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \frac{\partial^2 f}{\partial x_1 \partial x_3} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \frac{\partial^2 f}{\partial x_2 \partial x_3} \\ \frac{\partial^2 f}{\partial x_3 \partial x_1} & \frac{\partial^2 f}{\partial x_3 \partial x_2} & \frac{\partial^2 f}{\partial x_3^2} \end{bmatrix}
$$

## 7 Rank

矩阵的秩的定义

行秩（Row Rank）：矩阵中线性无关的行的最大数量。

列秩（Column Rank）：矩阵中线性无关的列的最大数量。

行秩和列秩是相等的，因此通常简称为矩阵的秩。

计算方法 计算矩阵的秩可以通过以下几种常用方法：

行简化法： 使用初等行变换将矩阵化为行简化阶梯形（Row Echelon Form）或标准行简化阶梯形（Reduced Row Echelon Form）。 非零行的数量即为矩阵的秩。 列简化法： 通过初等列变换将矩阵化为列简化阶梯形，非零列的数量对应于矩阵的秩。

列简化梯形形式：

$$
\begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{bmatrix}
$$

$$
\begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
$$

除了第一列，其他列都是零列，所以秩为1

列简化梯形要求：

* 矩阵的第一个非零行的第一个非零元素为1
* 每一行的第一个非零元素下面的元素都为0

Null Space

* $Ax=0$的解空间称为$A$的零空间

上三角矩阵：

* 主对角线以下的元素都是0

$$
\begin{bmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{bmatrix}
$$

下三角矩阵：

* 主对角线以上的元素都是0

$$
\begin{bmatrix} 1 & 0 & 0 \\ 2 & 3 & 0 \\ 4 & 5 & 6 \end{bmatrix}
$$

## 8 Trace

矩阵的迹是矩阵主对角线上元素的和，记作$tr(A)$

$$
tr(A) = \Sigma_{i=1}^{n}a_{i,i}
$$

迹的性质：

* $tr(A+B) = tr(A) + tr(B)$
* $tr(AB) = tr(BA)$
