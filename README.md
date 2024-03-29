# Python for Linear Algebra and Its Applications
## 利用Python学习线性代数
* 《线性代数及其应用》(David C. Lay) 一书中图示、算法、实例、习题等的 Python 实现。

线性代数是现代图像处理、数值算法、最优化、机器学习等的数学基础。
学懂线性代数，并把它作为一种数学语言，熟练的掌握它，应用它，在相关工程和研究工作中十分重要。

要掌握线性代数，仅仅看书往往是不够的，常见的问题有以下几种：
* 关键概念理解有偏差而不自知
* 理解不深刻
* 学完即忘
* 工作中不会实用

通过Learning by doing的方式，写代码实现其中的关键算法，测试关键实例，解决重要习题，往往能加深对抽象概念的理解，发现理解上的错误，形成更好的数学直觉。
写代码实现的算法和实例多了，还可以**缩小手眼差距**，避免一说都懂，一做就错，不会触类旁通的尴尬。
坚持**理论学习和编程动手实践交错进行**，互相补充，是深入理解线性代数，避免出现上述问题的好方法。

这个笔记是对应《线性代数及其应用》(David C. Lay) 一书第5版的 Python 代码，计划覆盖其中各个章节的主要算法、实例、图示和重要习题等。
在算法实现方面，笔记中大部分采用Numpy的ndarray作为基本的向量和矩阵数据结构，但是在学习向量和矩阵乘法等基础概念时，为演示细节，会直接用Python的列表和元组等基本类型。
在图例可视化方面，笔记中借助matplotlib库，绘制书中重要的点线面相关的图。

笔记的章节安排与书中一致，主要内容用Notebook的形式，将文字、公式、代码和结果按顺序呈现。
Notebook中的主要算法，独立形成Python源文件。
笔记中的代码风格尽量符合PEP 8的要求，算法实现不以效率为目标进行优化，而尽量以贴合书中概念为目标。

建议先阅读书本，然后参考学习代码实现，再回顾书本。

## 目录
1. 线性代数中的线性方程组
    1. [线性方程组](./chapter.01/section.1.01.notes.ipynb)
    2. 行化简与阶梯形矩阵
    3. [向量方程](./chapter.01/section.1.03.notes.ipynb)
2. 矩阵代数

## 代码运行环境
本笔记代码在Python 3.6 解释器运行通过，更高的Python版本应当兼容，但部分代码不兼容Python 2.

运行代码需安装以下依赖库：Numpy、Matplotlib、Jupyter。
```
pip install jupyter notebook numpy matplotlib -U
```

查看本笔记或跟随笔记试验代码，建议使用微软开源的编辑器 [vs code](https://code.visualstudio.com/)，并安装[Python扩展](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
