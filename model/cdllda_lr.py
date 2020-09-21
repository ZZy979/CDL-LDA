import logging
from collections import Counter

import numpy as np
from numba import jit
from sklearn.linear_model import LogisticRegression

from model.cdllda import CdlLdaModel, Domain


logger = logging.getLogger(__name__)


class CdlLdaLRModel(CdlLdaModel):
    """嵌入逻辑回归的CDL-LDA模型"""
    name = 'CDL-LDA-LR'

    def __init__(self, corpus, id2word, iterations=40, update_every=8, gibbs_iter=4,
                 n_groups=2, n_topics_c=6, n_topics_s=6, alpha=10.0, beta=0.1,
                 gamma_c=1000.0, gamma_s=1000.0, eta=0.01, seed=45):
        """构造一个CDL-LDA-LR模型

        :param corpus: iterable of corpora.Document，语料库
        :param id2word: 字典{单词id: 单词}
        :param iterations: 迭代次数
        :param update_every: 每迭代多少次更新参数
        :param gibbs_iter: 吉布斯采样迭代次数
        :param n_groups: 主题组数量
        :param n_topics_c: 公共主题数量
        :param n_topics_s: 特有主题数量
        :param alpha: 文档-主题分布θ的Dirichlet先验
        :param beta: 主题-单词分布φ的Dirichlet先验
        :param gamma_c: 公共主题类型分布σ的Beta先验
        :param gamma_s: 特有主题类型分布σ的Beta先验
        :param eta: 主题组分布π的Dirichlet先验
        :param seed: 随机数种子
        """
        self._n_groups = n_groups
        super().__init__(
            corpus, id2word, iterations, update_every, n_topics_c, n_topics_s,
            alpha, beta, gamma_c, gamma_s, eta, seed
        )
        self.gibbs_iter = gibbs_iter
        # 文档中单词的主题组频率，D*G
        self.topic_group_freq = np.zeros((len(self.corpus), n_groups))
        # 逻辑回归模型
        self.lr_model = LogisticRegression()
        # 规范化的逻辑回归模型参数
        self.psi = np.zeros((1, n_groups))
        logger.info('n_groups = %d, gibbs_iter = %d', n_groups, gibbs_iter)

    @property
    def n_groups(self):
        return self._n_groups

    def init(self):
        super().init()
        self.train_lr()

    def init_one_word(self, doc):
        _, r, z = super().init_one_word(doc)
        g = np.random.randint(self.n_groups)
        return g, r, z

    def train_lr(self):
        """训练逻辑回归模型并更新其模型参数psi。"""
        for d in range(len(self.corpus)):
            counter = Counter(self.topic_group[d])
            self.topic_group_freq[d] = \
                np.array([counter[g] for g in range(self.n_groups)]) / self.n_word_by_doc[d]
        # 以源域文档的主题组频率为特征，维数为G
        X = self.topic_group_freq[:self.corpus.n_source]
        y = np.array([doc.label for doc in self.corpus if doc.domain == Domain.SOURCE])
        self.lr_model = LogisticRegression(penalty='l2', tol=1e-2, C=1.0, max_iter=1000).fit(X, y)
        self.psi = self.lr_model.coef_ / np.linalg.norm(self.lr_model.coef_)

    def sample(self):
        for i in range(self.gibbs_iter):
            super().sample()
        self.train_lr()

    def predict_label(self):
        return self.lr_model.predict(self.topic_group_freq[self.corpus.n_source:])
