
### Week 1: 深度学习概论
#### What is a Neural Nerwork
linear regress 线性回归
ReLU function 修正线性单元(rectified linear unit)
修正：取非负数
若干个输入值x（输入层）经过由若干个ReLu形成的神经网络最终得到预期值y
#### Supervised Learning with Neural Networks 用神经网络进行监督学习
这是一种机器学习，目前因此获利最大为online advertising
>![0c6a840c37d141193e7af4aaedc07e65.png](en-resource://database/2871:1)
Some examples of Supervised Learning
1. Standard Neural Network
2. Convolutional Neural Networks 卷积神经网络: image data
3. Recurrent Neural Network循环神经网络: sequence data (audio has a temporal component时间成分，音频是随着时间播放的——one-dimensional time series一维时间序列 && 语言字母单词都是逐个出现的)
4. Hybrid Neural Network architecture
![ab380bd3122e23e64f0ca9f90b6684f2.png](en-resource://database/2873:1)
结构化数据：每个特征都有着清晰的定义
![e66743c505a20bfc5fcaaf3e7ed7842d.png](en-resource://database/2875:1)
#### 为什么深度学习会兴起
**Scale drives deep learning progress**
1. 需要训练一个规模足够大的神经网络（有许多hidden units, pareneters, connections）
2. x的值需要一定的体量即需要很多数据 (scale of the data)
![783c10ff92a1862b672992b8509b284d.png](en-resource://database/2879:1)
m: size of my training sets 训练集的规模
amout of labeled data 带标签的数据在训练样本时，输入x和标签y
**Sigmoid function**
When you implement gradient descent and gradient is 0, the parameters just change very slowly and so learing becomes really slow. 
**-> ReLU function**
Wheras by chaning the what's called the activation function 改变激活参数
其梯度对于所有为正值的输入输出都是1（梯度不会逐渐趋向0）
**使得 梯度下降法 运行得更快**
算法创新的结果：增加计算速度 || 在合理的时间内完成运算
在实现神经网络时，迭代速度对你的效率影响巨大。
![972c89b1ccd735c1dcaf07e67a49a51c.png](en-resource://database/2881:1)

### Week 2: Basic of Neural Network Programming
#### Binary Claasification 二分分类
![eef87ca7357faf9d25a979e7e7a5f455.png](en-resource://database/2887:1)

#### Logistic Regression
![61f8331112f774039d5b698b72d0ed9c.png](en-resource://database/2885:1)
sigmoid(z)
w和参数b分开，这里b对应一个拦截器
定义一个成本函数

#### Logistic Regression cost fuction
**损失函数：衡量算法的运行情况**
**在单个训练样本中定义的，衡量在单个训练样本上的表现**
**定义一个起着与误差平方相似的作用的误差函数**
ŷ是由simoid函数得出的，永远不会比1大
损失函数足够小，则log(1-ŷ)足够大
![265c6f6683f9698daefb3eaa243ec3dc.png](en-resource://database/2891:1)
当y=0时(0<ŷ<1)，损失函数会让ŷ->0（足够小）
当y=1时，ŷ尽可能大
**成本函数：衡量在全体训练样本上的表现**
**参数w和b在训练集上的效果**
ŷ：用作一组特定的参数w和b，通过logistic回归算法得出预测输出值
在训练logistic回归模型时，需要找到合适的参数w和b，让成本函数J尽可能的小
#### Gradient Descent
![ae662e3e1b445e7f634f2d840fe66388.png](en-resource://database/2895:1)
成本函数J(w, b)是在水平轴w和b上的曲面
曲面的高度：J(w, b)在某一点的值
凸函数无论在哪里初始化都应该达到同一点或大致相同的点
从初始点开始朝最陡的下坡方向走一步，在梯度下降一步后，或许就在那里停下（正试图沿着最快下降的方向往下走）——梯度下降的我一次迭代
最终收敛到这个全局的最优解 || 接近全局的最优解
![7f23cadda2722269abb851e08179329d.png](en-resource://database/2897:1)
α：学习率，可以控制每一次迭代 或者 梯度下降法中的步长
dw：导数（的变量名)，对参数w的更新（变化量）(函数在此点的斜率）

#### Computation Graph
Compute derivatives 计算神经网络的输出紧接着进行一个反向传输操作

#### Derivatives with a Computation Graph

#### Logistic Regression Gradient descent
![a9915e88ef9f17a35cf4fa45f082f7c7.png](en-resource://database/2903:1)
偏导数流程图
只考虑单个样本的情况
a is the output of the logistic regression
y is the ground truth label 样本的基本真值标签值
![db963ec25efc0d3f0ebdee605b3b9a9a.png](en-resource://database/2905:1)
链式法则
 分别计算dz, dw1, dw2, db
 alpha: learning rate
 单个样本实例的一次梯度更新步骤
 
 #### Gradient descent on m examples
 The overall cost functions with the sum: the average of the 1 over m term of the individual losses （1到m项损失函数和的平均）
 ![62330ccf751dbded0c360a9877850bb3.png](en-resource://database/2982:1)
 全局成本函数对w1的导数 <=> 各项损失函数对w1导数的平均
 ![4326380c538368776f8bab491f3d4d33.png](en-resource://database/2984:1)
 Use dw1, dw2 and db as accumulators
 dw1 = 全局成本函数对w1的导数
 dw1, dw2没有上标i，使用它们作为累加器去求取整个训练集上上的和
 for：遍历m个训练样本的小循环，遍历所有特征的for循环 算法低效
 向量化技术vectorization：摆脱for循环