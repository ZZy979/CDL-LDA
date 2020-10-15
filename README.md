# Cross-Domain Labeled LDA (CDL-LDA)
原论文：<https://arxiv.org/pdf/1809.05820.pdf>

## [CDL-LDA](model/cdllda.py)
原论文模型

## [CDL-LDA-un](model/cdllda_un.py)
原论文中的无监督版本的CDL-LDA模型
* 初始化过程中源域文档单词的主题组不使用文档的标签，而是随机生成
* 预测文档标签使用逻辑回归而不是文档中单词的主题组

## [CDL-LDA-soft](model/cdllda_soft.py)
使用soft prior的CDL-LDA模型
* 单词主题组可以使用soft prior

## [CDL-LDA-LR](model/cdllda_lr.py)
嵌入逻辑回归的CDL-LDA模型
* 单词的主题组不再对应文档标签，主题组的数量和标签数量可以不同
* 生成过程嵌入逻辑回归模型，和模型本身同时训练
* 预测文档标签使用训练得到的逻辑回归模型
