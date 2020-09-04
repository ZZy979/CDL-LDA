"""
Cross-Domain Labeled LDA (CDL-LDA) <https://arxiv.org/pdf/1809.05820.pdf>
Python reimplementation.
"""
import logging
import time
from enum import IntEnum

import numpy as np
from gensim import interfaces
from gensim.models import basemodel
from tqdm import tqdm

from corpora import Corpus


class Domain(IntEnum):
    """文档所属的域：源/目标"""
    SOURCE = 0
    TARGET = 1


class TopicType(IntEnum):
    """主题类型：公共/特有"""
    COMMON = 0
    SPECIFIC = 1


logger = logging.getLogger(__name__)


class CdlLdaModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """标准CDL-LDA模型"""

    def __init__(self, corpus: Corpus, iterations=50, update_every=10,
                 n_topics_c=6, n_topics_s=6, alpha=10.0, beta=0.1,
                 gamma_c=1000.0, gamma_s=1000.0, eta=0.01, seed=90):
        """构造一个CDL-LDA模型

        :param corpus: corpora.Corpus，语料库
        :param iterations: 迭代次数
        :param update_every: 每迭代多少次更新参数
        :param n_topics_c: 公共主题数量
        :param n_topics_s: 特有主题数量
        :param alpha: 文档-主题分布θ的Dirichlet先验
        :param beta: 主题-单词分布φ的Dirichlet先验
        :param gamma_c: 公共主题类型分布σ的Beta先验
        :param gamma_s: 特有主题类型分布σ的Beta先验
        :param eta: 标签（主题组）分布π的Dirichlet先验
        :param seed: 随机数种子
        """
        # ----------语料库----------
        self.corpus = corpus

        # ----------模型参数----------
        self.iterations = iterations
        self.update_every = update_every
        self.n_topics_c = n_topics_c
        self.n_topics_s = n_topics_s
        self.alpha = alpha
        self.beta = beta
        self.gamma_c = gamma_c
        self.gamma_s = gamma_s
        self.gamma = np.array([gamma_c, gamma_s])
        self.eta = eta
        np.random.seed(seed)

        # ----------各种维度的单词数统计----------
        # V - 单词表大小
        # D - 文档数
        # L - 标签数
        # R - 主题类型数（公共/特有），=2
        # M - 域数（源/目标），=2
        # TC - 公共主题数
        # TS - 特有主题数
        # -------------------------------------
        V = self.corpus.n_vocabs
        D = len(self.corpus)
        L = self.corpus.n_labels
        R = len(TopicType.__members__)
        M = len(Domain.__members__)
        # 文档单词数，D
        self.n_word_by_doc = np.zeros(D, dtype=np.int)
        # 每个文档中每个单词的出现次数，D*V
        self.count_word_by_doc = np.zeros((D, V), dtype=np.int)
        # 每个文档中每个标签的出现次数，D*L
        self.count_label_by_doc = np.zeros((D, L), dtype=np.int)
        # 每个文档、标签、主题类型对应的单词数，D*L*R
        self.n_word_by_doc_label_type = np.zeros((D, L, R), dtype=np.int)
        # 每个文档、标签、公共主题对应的单词数，D*L*TC
        self.n_word_by_doc_label_topic_c = np.zeros((D, L, self.n_topics_c), dtype=np.int)
        # 每个文档、标签、特有主题对应的单词数，D*L*TS
        self.n_word_by_doc_label_topic_s = np.zeros((D, L, self.n_topics_s), dtype=np.int)
        # 每个标签、公共主题下每个单词的出现次数，L*TC*V
        self.count_word_by_label_topic_c = np.zeros((L, self.n_topics_c, V), dtype=np.int)
        # 每个域、标签、特有主题下每个单词的出现次数，M*L*TS*V
        self.count_word_by_domain_label_topic_s = np.zeros((M, L, self.n_topics_s, V), dtype=np.int)
        # 每个标签、公共主题对应的单词数，L*TC
        self.n_word_by_label_topic_c = np.zeros((L, self.n_topics_c), dtype=np.int)
        # 每个域、标签、特有主题对应的单词数，M*L*TS
        self.n_word_by_domain_label_topic_s = np.zeros((M, L, self.n_topics_s), dtype=np.int)

        # ----------隐变量：每个单词的标签（主题组）、主题类型、主题----------
        # 每个文档中每个单词的标签（主题组），D*Nd
        self.label = [[] for d in range(D)]
        # 每个文档中每个单词的主题类型，D*Nd
        self.topic_type = [[] for d in range(D)]
        # 每个文档中每个单词的主题，D*Nd
        self.topic = [[] for d in range(D)]

        # ----------分布参数（用于累加？）----------
        # 公共主题-单词分布，L*TC*V
        self.phi_c = np.zeros((L, self.n_topics_c, V))
        # 特有主题-单词分布，M*L*TS*V
        self.phi_s = np.zeros((M, L, self.n_topics_s, V))
        # 文档-公共主题分布，D*L*TC
        self.theta_c = np.zeros((D, L, self.n_topics_c))
        # 文档-特有主题分布，D*L*TS
        self.theta_s = np.zeros((D, L, self.n_topics_s))
        # 主题类型分布，D*L*R
        self.sigma = np.zeros((D, L, R))
        # 标签（主题组）分布，D*L
        self.pi = np.zeros((D, L))
        # 参数更新次数
        self.update_count = 0

        self.init()

    def init(self):
        """根据语料库初始化单词数统计、隐变量和分布参数。"""
        # d - 文档编号
        # w - 单词在单词表中的编号
        # i - 单词在文档中的索引
        # m - 域
        # l - 标签/主题分组编号
        # r - 主题类型
        # z - 主题编号
        logger.info('Initializing...')
        L = self.corpus.n_labels
        for d, doc in enumerate(self.corpus):
            self.n_word_by_doc[d] = len(doc)
            self.label[d] = [0] * len(doc)
            self.topic_type[d] = [0] * len(doc)
            self.topic[d] = [0] * len(doc)
            for i, w in enumerate(doc):
                self.count_word_by_doc[d, w] += 1
                m = doc.domain
                # TODO 标签l应从多项式分布pi中采样，主题类型r应从伯努利分布sigma中采样
                # TODO 文档-主题分布theta和主题-单词分布phi应从狄利克雷分布alpha, beta中采样
                self.label[d][i] = l = doc.label if m == Domain.SOURCE \
                    else np.random.randint(L)
                self.count_label_by_doc[d, l] += 1

                self.topic_type[d][i] = r = np.random.choice((TopicType.COMMON, TopicType.SPECIFIC))
                if r == TopicType.COMMON:
                    self.topic[d][i] = z = np.random.randint(self.n_topics_c)
                    self.n_word_by_doc_label_topic_c[d, l, z] += 1
                    self.count_word_by_label_topic_c[l, z, w] += 1
                    self.n_word_by_label_topic_c[l, z] += 1
                    self.n_word_by_doc_label_type[d, l, r] += 1
                else:
                    self.topic[d][i] = z = np.random.randint(self.n_topics_s)
                    self.n_word_by_doc_label_topic_s[d, l, z] += 1
                    self.count_word_by_domain_label_topic_s[m, l, z, w] += 1
                    self.n_word_by_domain_label_topic_s[m, l, z] += 1
                    self.n_word_by_doc_label_type[d, l, r] += 1
        logger.info('Initialization finished.')

    def train(self):
        """训练模型"""
        logger.info(
            'Training CDL-LDA on %s dataset, %d documents, %d iterations, update every %d, '
            '%d common topics, %d specific topics, alpha = %s, beta = %s, '
            'gamma_c = %s, gamma_s = %s, eta = %s',
            self.corpus.name, len(self.corpus), self.iterations, self.update_every,
            self.n_topics_c, self.n_topics_s, self.alpha, self.beta,
            self.gamma_c, self.gamma_s, self.eta
        )
        for it in range(self.iterations):
            start_time = time.time()
            for d, doc in enumerate(tqdm(self.corpus)):
                m = doc.domain
                for i, w in enumerate(doc):
                    l, r, z = self.label[d][i], self.topic_type[d][i], self.topic[d][i]
                    self.update_word_count(d, m, l, r, z, w, -1)
                    l, r, z = self.do_estep(d, m, w)
                    self.label[d][i], self.topic_type[d][i], self.topic[d][i] = l, r, z
                    self.update_word_count(d, m, l, r, z, w, 1)
            elapsed = time.time() - start_time
            if (it + 1) % self.update_every == 0 or it == self.iterations - 1:
                logger.info('Iteration %d/%d, time: %.2f s', it + 1, self.iterations, elapsed)
                self.do_mstep()
        logger.info('Training finished.')

    def do_estep(self, d, m, w):
        """E步：采样生成单词w的标签、主题类型和主题。

        遍历所有的标签l、主题类型r和主题z，计算P(l,r,z,w|d)，以此为概率进行采样。

        :param d: 单词w所属文档编号
        :param m: 文档d所属的域
        :param w: 目标单词在单词表中的编号
        :return: 标签l、主题类型r、主题z
        """
        #          ┌────────┬──────────┬────────┬──────────┐
        # p_table: │ common │ specific │ common │ specific │
        #          └────────┴──────────┴────────┴──────────┘
        #          │      l = 0        |      l = 1        │
        period = self.n_topics_c + self.n_topics_s
        p_table = self.calc_joint_distribution(d, m, w).ravel()
        index = np.random.choice(p_table.size, p=p_table)
        l = index // period
        offset = index - l * period
        if offset < self.n_topics_c:
            r = TopicType.COMMON
            z = offset
        else:
            r = TopicType.SPECIFIC
            z = offset - self.n_topics_c
        return l, r, z

    def update_word_count(self, d, m, l, r, z, w, delta):
        """更新属于文档d、域m、标签l、主题类型r、主题z的单词w的计数，delta = 1/-1。"""
        self.count_label_by_doc[d, l] += delta
        self.n_word_by_doc_label_type[d, l, r] += delta
        if r == TopicType.COMMON:
            self.count_word_by_label_topic_c[l, z, w] += delta
            self.n_word_by_label_topic_c[l, z] += delta
            self.n_word_by_doc_label_topic_c[d, l, z] += delta
        else:
            self.count_word_by_domain_label_topic_s[m, l, z, w] += delta
            self.n_word_by_domain_label_topic_s[m, l, z] += delta
            self.n_word_by_doc_label_topic_s[d, l, z] += delta

    def calc_joint_distribution(self, d, m, w):
        """计算文档d中的单词w的标签-主题联合概率分布。

        :param d: 单词w所属文档编号
        :param m: 文档d所属的域
        :param w: 目标单词在单词表中的编号
        :return: ndarray(L, TC + TS), 0~TC-1列表示公共主题，TC~TC+TS-1列表示特有主题
        """
        distribution = np.ones((self.corpus.n_labels, self.n_topics_c + self.n_topics_s))
        distribution *= self.calc_phi(m, w)
        distribution *= self.calc_theta(d)
        sigma = self.calc_sigma(d)
        distribution[:, :self.n_topics_c] *= sigma[:, TopicType.COMMON][:, np.newaxis]
        distribution[:, self.n_topics_c:] *= sigma[:, TopicType.SPECIFIC][:, np.newaxis]
        distribution *= self.calc_pi(d, m)[:, np.newaxis]
        distribution /= distribution.sum()
        return distribution

    def calc_phi(self, m, w):
        """计算属于域m、标签l的主题z产生单词w的概率。

        :return: ndarray(L, TC + TS), 0~TC-1列表示公共主题，TC~TC+TS-1列表示特有主题
        """
        phi = np.zeros((self.corpus.n_labels, self.n_topics_c + self.n_topics_s))
        phi[:, :self.n_topics_c] = (self.count_word_by_label_topic_c[:, :, w] + self.beta) \
                / (self.n_word_by_label_topic_c + self.corpus.n_vocabs * self.beta)
        phi[:, self.n_topics_c:] = (self.count_word_by_domain_label_topic_s[m, :, :, w] + self.beta) \
                / (self.n_word_by_domain_label_topic_s[m] + self.corpus.n_vocabs * self.beta)
        return phi

    def calc_theta(self, d):
        """计算文档d的标签-主题联合分布。

        :return: ndarray(L, TC + TS), 0~TC-1列表示公共主题，TC~TC+TS-1列表示特有主题
        """
        theta = np.zeros((self.corpus.n_labels, self.n_topics_c + self.n_topics_s))
        theta[:, :self.n_topics_c] = (self.n_word_by_doc_label_topic_c[d] + self.alpha) \
                  / (self.n_word_by_doc_label_type[d, :, TopicType.COMMON][:, np.newaxis]
                     + self.n_topics_c * self.alpha)
        theta[:, self.n_topics_c:] = (self.n_word_by_doc_label_topic_s[d] + self.alpha) \
                  / (self.n_word_by_doc_label_type[d, :, TopicType.SPECIFIC][:, np.newaxis]
                     + self.n_topics_s * self.alpha)
        return theta

    def calc_sigma(self, d):
        """计算文档d的标签-主题类型联合分布。

        :return: ndarray(L, R), p[l][r]表示文档d的标签为l、主题类型为r的概率
        """
        return (self.n_word_by_doc_label_type[d] + self.gamma) \
               / (self.count_label_by_doc[d][:, np.newaxis] + self.gamma.sum())

    def calc_pi(self, d, m):
        """计算属于域m的文档d的标签概率分布。

        :return: ndarray(L)，第l个元素表示d属于标签l的概率
        """
        if m == Domain.SOURCE:
            return self.count_label_by_doc[d] / self.n_word_by_doc[d]
        else:
            return (self.count_label_by_doc[d] + self.eta) \
                   / (self.n_word_by_doc[d] + self.corpus.n_labels * self.eta)

    def do_mstep(self):
        """M步：更新分布参数。"""
        self.phi_c *= self.update_count
        self.phi_s *= self.update_count
        self.theta_c *= self.update_count
        self.theta_s *= self.update_count
        self.sigma *= self.update_count
        self.pi *= self.update_count

        self.phi_c += (self.count_word_by_label_topic_c + self.beta) \
                      / (self.n_word_by_label_topic_c[:, :, np.newaxis]
                         + self.corpus.n_vocabs * self.beta)
        self.phi_s += (self.count_word_by_domain_label_topic_s + self.beta) \
                      / (self.n_word_by_domain_label_topic_s[:, :, :, np.newaxis]
                         + self.corpus.n_vocabs * self.beta)
        self.theta_c += (self.n_word_by_doc_label_topic_c + self.alpha) \
                        / (self.n_word_by_doc_label_type[:, :, TopicType.COMMON][:, :, np.newaxis]
                           + self.n_topics_c * self.alpha)
        self.theta_s += (self.n_word_by_doc_label_topic_s + self.alpha) \
                        / (self.n_word_by_doc_label_type[:, :, TopicType.SPECIFIC][:, :, np.newaxis]
                           + self.n_topics_s * self.alpha)
        self.sigma += (self.n_word_by_doc_label_type + self.gamma) \
                      / (self.count_label_by_doc[:, np.newaxis] + self.gamma.sum())
        for d, doc in enumerate(self.corpus):
            self.pi[d] += self.calc_pi(d, doc.domain)

        self.update_count += 1
        self.phi_c /= self.update_count
        self.phi_s /= self.update_count
        self.theta_c /= self.update_count
        self.theta_s /= self.update_count
        self.sigma /= self.update_count
        self.pi /= self.update_count

    def predict_label(self):
        """预测目标域文档的标签，选择一篇文档中出现次数最多的单词标签作为该文档的标签。

        :return: ndarray(Dt), Dt为目标域文档数
        """
        return np.array([
            self.count_label_by_doc[d].argmax()
            for d, doc in enumerate(self.corpus) if doc.domain == Domain.TARGET
        ])

    def calc_perplexity(self):
        r"""计算模型的困惑度。

        .. math::
            perplexity = \exp (- \frac{\sum_{d \in {D^t}} {\log P(d)}}{{\sum_{d \in {D^t}} {N_d}}})

            \log P(d) = \sum_{w \in d} {({N_{d,w}}\log \sum_{l \in L} {\sum_{r \in R}
             {\sum_{z \in {T^r}} {\theta _{d,l}^r(z)\phi _{Target,l,z}^r(w)}}})}
        """
        sum_logp = 0.0
        for d, doc in enumerate(self.corpus):
            if doc.domain == Domain.TARGET:
                logp = 0.0
                for w in range(self.corpus.n_vocabs):
                    if self.count_word_by_doc[d, w] == 0:
                        continue
                    s = np.sum(self.theta_c[d, :, :] * self.phi_c[:, :, w]
                               + self.theta_s[d, :, :] * self.phi_s[Domain.TARGET, :, :, w])
                    if s > 0:
                        logp += self.count_word_by_doc[d, w] * np.log(s)
                sum_logp += logp
        sum_len = sum(len(doc) for doc in self.corpus if doc.domain == Domain.TARGET)
        return np.exp(-sum_logp / sum_len)

    def __getitem__(self, vec):
        # TODO 预测新文档？
        pass

    def get_topics(self):
        pass
