import logging

import numpy as np

from model.cdllda import CdlLdaModel, Domain

logger = logging.getLogger(__name__)


class CdlLdaSoftModel(CdlLdaModel):
    """使用soft prior的CDL-LDA模型

    单词主题组可以使用soft prior
    """
    name = 'CDL-LDA-soft'

    def __init__(self, corpus, id2word, iterations=40, update_every=8,
                 n_topics_c=6, n_topics_s=6, alpha=10.0, beta=0.01,
                 gamma_c=1000.0, gamma_s=1000.0, eta=0.1, seed=45, use_soft=False, prior=None):
        """构造一个CDL-LDA-soft模型

        :param corpus: iterable of corpora.Document，语料库
        :param id2word: 字典{单词id: 单词}
        :param iterations: 迭代次数
        :param update_every: 每迭代多少次更新参数
        :param n_topics_c: 公共主题数量
        :param n_topics_s: 特有主题数量
        :param alpha: 文档-主题分布θ的Dirichlet先验
        :param beta: 主题-单词分布φ的Dirichlet先验
        :param gamma_c: 公共主题类型分布σ的Beta先验
        :param gamma_s: 特有主题类型分布σ的Beta先验
        :param eta: 主题组分布π的Dirichlet先验
        :param seed: 随机数种子
        :param use_soft: 单词主题组是否使用soft prior
        :param prior: ndarray(V, G)，单词主题组先验概率
        """
        super().__init__(
            corpus, id2word, iterations, update_every, n_topics_c, n_topics_s,
            alpha, beta, gamma_c, gamma_s, eta, seed
        )
        self.use_soft = use_soft
        self.prior = prior
        logger.info('use soft prior = %s', use_soft)

    def calc_pi(self, d, m):
        if self.use_soft and m == Domain.SOURCE:
            return _calc_pi_soft(np.array(self.corpus[d], dtype=np.int32), self.prior)
        else:
            return super().calc_pi(d, m)


def _calc_pi_soft(doc, prior):
    pi = np.sum(prior.take(doc, axis=0), axis=0)
    return pi / pi.sum()
