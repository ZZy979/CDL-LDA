import logging

import numpy as np

from model.cdllda import CdlLdaModel, TopicType, Domain

logger = logging.getLogger(__name__)


class CdlLdaSoftModel(CdlLdaModel):
    """改进版本的CDL-LDA模型

    1. 单词标签可以使用soft prior
    2. 源域和目标域的特有主题数量可以不同
    """
    name = 'CDL-LDA-soft'

    def __init__(self, corpus, id2word, iterations=40, update_every=8,
                 n_topics_c=6, n_topics_s_src=6, n_topics_s_tgt=6, alpha=10.0, beta=0.1,
                 gamma_c=1000.0, gamma_s=1000.0, eta=0.01, seed=45, use_soft=False, prior=None):
        """构造一个CDL-LDA-soft模型

        :param corpus: iterable of corpora.Document，语料库
        :param id2word: 字典{单词id: 单词}
        :param iterations: 迭代次数
        :param update_every: 每迭代多少次更新参数
        :param n_topics_c: 公共主题数量
        :param n_topics_s_src: 源域特有主题数量
        :param n_topics_s_tgt: 目标域特有主题数量
        :param alpha: 文档-主题分布θ的Dirichlet先验
        :param beta: 主题-单词分布φ的Dirichlet先验
        :param gamma_c: 公共主题类型分布σ的Beta先验
        :param gamma_s: 特有主题类型分布σ的Beta先验
        :param eta: 标签（主题组）分布π的Dirichlet先验
        :param seed: 随机数种子
        :param use_soft: 单词标签是否使用soft prior
        :param prior: ndarray(V, L)，单词标签先验概率
        """
        super().__init__(
            corpus, id2word, iterations, update_every, n_topics_c,
            max(n_topics_s_src, n_topics_s_tgt), alpha, beta, gamma_c, gamma_s, eta, seed
        )
        self._n_topics_s = np.array([n_topics_s_src, n_topics_s_tgt])
        self.use_soft = use_soft
        self.prior = prior
        logger.info(
            '%d specific topics in source domain, %d in target, use soft prior = %s',
            n_topics_s_src, n_topics_s_tgt, use_soft
        )

    def init_one_word(self, doc):
        l, r, z = super().init_one_word(doc)
        if r == TopicType.SPECIFIC:
            z = np.random.randint(self._n_topics_s[doc.domain])
        return l, r, z

    def calc_phi(self, m, w):
        # TS = max{TS(src), TS(tgt)}，若TS(src) < TS(tgt)则TS(src):TS对应的列要清零，对TS(tgt)同理
        phi = super().calc_phi(m, w)
        phi[:, self.n_topics_c + self._n_topics_s[m]:self.n_topics_c + self.n_topics_s] = 0
        return phi

    def calc_theta(self, d):
        # 超类的calc_theta()不需要参数m，但该类区分源域和目标域的特有主题数就需要，只能给Corpus类增加索引访问
        theta = super().calc_theta(d)
        m = self.corpus[d].domain
        theta[:, self.n_topics_c + self._n_topics_s[m]:self.n_topics_c + self.n_topics_s] = 0
        return theta

    def calc_pi(self, d, m):
        if self.use_soft and m == Domain.SOURCE:
            return _calc_pi_soft(d, np.array(self.corpus[d], dtype=np.int32), self.prior)
        else:
            return super().calc_pi(d, m)


def _calc_pi_soft(d, doc, prior):
    pi = np.sum(prior.take(doc, axis=0), axis=0)
    return pi / pi.sum()
