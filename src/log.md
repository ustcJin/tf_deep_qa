# 记录
这个主要记录这个project的进展，目前是实现了对theano版本的改写，中间很多曲折，主要收获点及结论总结如下，当然后面会马不停蹄的继续优化。
- 可用。做到了论文中的效果，目前预测的准确率:1217 / 1500 = 81.3%
- 语法糖:主要针对theano和tensorflow
	＊ 输出查看，theano使用eval()可直接查看，tensorflow查看Tensor变量需要Run，print sess.run(val), 才能查看val
	＊ 变量设置(调参), theano直接使用shared变量做更新，迭代一次之后计算grad，可观察的更新参数，而tensorflow需要设置成Variable形式(否则一直是初始值)，优化算法一般使用AdamOptimizer，不使用Adadelta，会自动对所有的Variable进行优化。
	＊ 观测:theano观测需要参数与观测部件的输入精确匹配，而tensorflow可以富裕参数
- 后面的方向:
	＊ 保存图到文件，load的方式，不能测试的时候一直构图吧，一种是使用cPickle(郁闷，没有成功)，还有一种是tf自带的存图方式，可以从简单的开始学起
	＊ 参数保存，load，较方便的方式，目前的加载后直接硬塞的方式太low了
	＊ 优化脚本之后，可以做真实问答相关的项目


# 进展
时隔一天，就把整个过程tf化了，通过tf.train.Saver, 将整个过程也层次化了，整个项目分为四个阶段
- layer.py: 定义各个组件，基本的类(可以认为是dp各个层的基本单元)
- graph.py: 构图，并将图存储在文件中，通过saver.export_meta_graph
- train.py: 训练参数的，构图阶段，没有训练参数，在这个地方来做的, 之后通过saver.save(sess, 'qa.params')保存参数
- qa_test.py: 验证准确率，我实验的正确率是`0.827949901121`，基本达到论文要求的精度


基本的训练集参考论文
