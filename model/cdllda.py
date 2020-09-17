"""
Cross-Domain Labeled LDA (CDL-LDA) <https://arxiv.org/pdf/1809.05820.pdf>
Python reimplementation.
"""
import logging
import time
from enum import IntEnum

import numpy as np
from gensim import matutils
from gensim.models import basemodel
from numba import jit

import utils


class Domain(IntEnum):
    """文档所属的域：源/目标"""
    SOURCE = 0
    TARGET = 1


class TopicType(IntEnum):
    """主题类型：公共/特有"""
    COMMON = 0
    SPECIFIC = 1


logger = logging.getLogger(__name__)


class CdlLdaModel(basemodel.BaseTopicModel):
    """标准CDL-LDA模型"""
    name = 'CDL-LDA'

    def __init__(self, corpus, id2word, iterations=40, update_every=8,
                 n_topics_c=6, n_topics_s=6, alpha=10.0, beta=0.1,
                 gamma_c=1000.0, gamma_s=1000.0, eta=0.01, seed=45):
        """构造一个CDL-LDA模型

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
        :param eta: 标签（主题组）分布π的Dirichlet先验
        :param seed: 随机数种子
        """
        logger.info(
            'Constructing %s model with %d common topics, %d specific topics, '
            'alpha = %s, beta = %s, gamma_c = %s, gamma_s = %s, eta = %s, seed = %d',
            self.name, n_topics_c, n_topics_s, alpha, beta, gamma_c, gamma_s, eta, seed
        )
        # ----------语料库、单词表----------
        self.corpus = corpus
        self.id2word = id2word

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
        utils.seed(seed)

        # ----------各种维度的单词数统计----------
        # V - 单词表大小
        # D - 文档数
        # L - 标签数
        # R - 主题类型数（公共/特有），=2
        # M - 域数（源/目标），=2
        # TC - 公共主题数
        # TS - 特有主题数
        # -------------------------------------
        V = len(self.id2word)
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

        # ----------分布参数----------
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
        for d, doc in enumerate(self.corpus):
            self.n_word_by_doc[d] = len(doc)
            self.label[d] = [0] * len(doc)
            self.topic_type[d] = [0] * len(doc)
            self.topic[d] = [0] * len(doc)
            for i, w in enumerate(doc):
                self.count_word_by_doc[d, w] += 1
                l, r, z = self.init_one_word(doc)
                self.label[d][i], self.topic_type[d][i], self.topic[d][i] = l, r, z
                self.update_word_count(d, doc.domain, l, r, z, w, 1)
        logger.info('Initialization finished')

    def init_one_word(self, doc):
        """初始化一个单词的标签、主题类型和主题。

        :param doc: 单词所属的文档
        :return: 单词的标签、主题类型和主题
        """
        # TODO 标签l应从多项式分布pi中采样，主题类型r应从伯努利分布sigma中采样
        # TODO 文档-主题分布theta和主题-单词分布phi应从狄利克雷分布alpha, beta中采样
        l = doc.label if doc.domain == Domain.SOURCE else np.random.randint(self.corpus.n_labels)
        r = TopicType.COMMON if np.random.random() < 0.5 else TopicType.SPECIFIC
        z = np.random.randint(self.n_topics_c) if r == TopicType.COMMON \
            else np.random.randint(self.n_topics_s)
        return l, r, z

    def estimate(self):
        """参数估计（训练模型）"""
        self.init()
        logger.info(
            'Training %s model on %s dataset, %d documents, %d iterations, update every %d iter',
            self.name, self.corpus.name, len(self.corpus), self.iterations, self.update_every,
        )
        start_time = time.time()
        for it in range(self.iterations):
            for d, doc in enumerate(self.corpus):
                m = doc.domain
                for i, w in enumerate(doc):
                    l, r, z = self.label[d][i], self.topic_type[d][i], self.topic[d][i]
                    self.update_word_count(d, m, l, r, z, w, -1)
                    l, r, z = self.sample(d, m, w)
                    self.label[d][i], self.topic_type[d][i], self.topic[d][i] = l, r, z
                    self.update_word_count(d, m, l, r, z, w, 1)
            if (it + 1) % self.update_every == 0 or it == self.iterations - 1:
                elapsed = time.time() - start_time
                logger.info('Iteration %d/%d, time: %.2f s', it + 1, self.iterations, elapsed)
                self.update_param()
        logger.info('Training finished')

    def sample(self, d, m, w):
        """采样生成单词w的标签、主题类型和主题。

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
        index = utils.choice(p_table)
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
        return _calc_joint_distribution(
            self.n_topics_c, self.calc_phi(m, w), self.calc_theta(d),
            self.calc_sigma(d), self.calc_pi(d, m)[:, np.newaxis]
        )

    def calc_phi(self, m, w):
        """计算属于域m、标签l的主题z产生单词w的概率。

        :return: ndarray(L, TC + TS), 0~TC-1列表示公共主题，TC~TC+TS-1列表示特有主题
        """
        return _calc_phi(
            int(m), w, self.corpus.n_labels, len(self.id2word), self.n_topics_c, self.n_topics_s,
            self.beta, self.count_word_by_label_topic_c, self.count_word_by_domain_label_topic_s,
            self.n_word_by_label_topic_c, self.n_word_by_domain_label_topic_s
        )

    def calc_theta(self, d):
        """计算文档d的标签-主题联合分布。

        :return: ndarray(L, TC + TS), 0~TC-1列表示公共主题，TC~TC+TS-1列表示特有主题
        """
        return _calc_theta(
            d, self.corpus.n_labels, self.n_topics_c, self.n_topics_s, self.alpha,
            self.n_word_by_doc_label_type,
            self.n_word_by_doc_label_topic_c, self.n_word_by_doc_label_topic_s
        )

    def calc_sigma(self, d):
        """计算文档d的标签-主题类型联合分布。

        :return: ndarray(L, R), p[l][r]表示文档d的标签为l、主题类型为r的概率
        """
        return _calc_sigma(d, self.gamma, self.count_label_by_doc, self.n_word_by_doc_label_type)

    def calc_pi(self, d, m):
        """计算属于域m的文档d的标签概率分布。

        :return: ndarray(L)，第l个元素表示d属于标签l的概率
        """
        return _calc_pi(
            d, int(m), self.corpus.n_labels, self.eta, self.n_word_by_doc, self.count_label_by_doc
        )

    def update_param(self):
        """更新分布参数。"""
        self.phi_c *= self.update_count
        self.phi_s *= self.update_count
        self.theta_c *= self.update_count
        self.theta_s *= self.update_count
        self.sigma *= self.update_count
        self.pi *= self.update_count

        self.phi_c += (self.count_word_by_label_topic_c + self.beta) \
                      / (self.n_word_by_label_topic_c[:, :, np.newaxis]
                         + len(self.id2word) * self.beta)
        self.phi_s += (self.count_word_by_domain_label_topic_s + self.beta) \
                      / (self.n_word_by_domain_label_topic_s[:, :, :, np.newaxis]
                         + len(self.id2word) * self.beta)
        self.theta_c += (self.n_word_by_doc_label_topic_c + self.alpha) \
                        / (self.n_word_by_doc_label_type[:, :, TopicType.COMMON][:, :, np.newaxis]
                           + self.n_topics_c * self.alpha)
        self.theta_s += (self.n_word_by_doc_label_topic_s + self.alpha) \
                        / (self.n_word_by_doc_label_type[:, :, TopicType.SPECIFIC][:, :, np.newaxis]
                           + self.n_topics_s * self.alpha)
        self.sigma += (self.n_word_by_doc_label_type + self.gamma) \
                      / (self.count_label_by_doc[:, :, np.newaxis] + self.gamma.sum())
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
        """预测目标域文档的标签。

        :return: ndarray(Dt), Dt为目标域文档数
        """
        # 有监督预测：选择一篇文档中出现次数最多的单词标签作为该文档的标签
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
        return _calc_perplexity(
            np.array([d for d, doc in enumerate(self.corpus) if doc.domain == Domain.TARGET]),
            self.count_word_by_doc, self.phi_c, self.phi_s, self.theta_c, self.theta_s
        )

    def get_topics(self, m=Domain.SOURCE, l=0, r=TopicType.COMMON):
        """返回主题-单词分布。

        :param m: 域，仅对特有主题有效
        :param l: 标签
        :param r: 主题类型
        :return: ndarray(Tr, V)
        """
        if r == TopicType.COMMON:
            return self.phi_c[l] / self.phi_c[l].sum(axis=1)[:, np.newaxis]
        else:
            return self.phi_s[m][l] / self.phi_s[m][l].sum(axis=1)[:, np.newaxis]

    def show_topic(self, topicid, topn=10, m=Domain.SOURCE, l=0, r=TopicType.COMMON):
        """返回指定域、标签和类型的主题topicid的Top N (单词,概率)表示。

        :return: list of (str, float)
        """
        topic = self.get_topics(m, l, r)[topicid]
        return [(self.id2word[w], topic[w]) for w in matutils.argsort(topic, topn, True)]

    def show_topics(self, num_topics=10, num_words=10, log=False,
                    m=Domain.SOURCE, l=0, r=TopicType.COMMON):
        """返回指定主题的Top N (单词,概率)表示。

        :return: list of (int, list of (str, float))
        """
        n_topics = self.n_topics_c if r == TopicType.COMMON else self.n_topics_s
        if num_topics < 0 or num_topics > n_topics:
            num_topics = n_topics
        topics = []
        for z in range(num_topics):
            topic = self.show_topic(z, num_words, m, l, r)
            topics.append((z, topic))
            if log:
                logger.info('%s topic #%d: %s', r.name, z, topic)
        return topics

    def print_topic(self, topicno, topn=10, m=Domain.SOURCE, l=0, r=TopicType.COMMON):
        return ' + '.join('%.3f*"%s"' % (v, k) for k, v in self.show_topic(topicno, topn, m, l, r))

    def print_topics(self, num_topics=20, num_words=10, m=Domain.SOURCE, l=0, r=TopicType.COMMON):
        return self.show_topics(num_topics, num_words, True, m, l, r)


@jit(nopython=True)
def _calc_joint_distribution(n_topics_c, phi, theta, sigma, pi):
    """计算单词的标签-主题联合概率分布。

    :param n_topics_c: 公共主题数
    :param phi: ndarray(L, TC + TS)
    :param theta: ndarray(L, TC + TS)
    :param sigma: ndarray(L, R)
    :param pi: ndarray(L, 1)
    :return: ndarray(L, TC + TS)
    """
    distribution = phi * theta
    distribution[:, :n_topics_c] *= sigma[:, :1]
    distribution[:, n_topics_c:] *= sigma[:, 1:]
    distribution *= pi
    distribution /= distribution.sum()
    return distribution


@jit(nopython=True)
def _calc_phi(m, w, n_labels, n_vocabs, n_topics_c, n_topics_s, beta,
              count_word_by_label_topic_c, count_word_by_domain_label_topic_s,
              n_word_by_label_topic_c, n_word_by_domain_label_topic_s):
    phi = np.zeros((n_labels, n_topics_c + n_topics_s))
    phi[:, :n_topics_c] = (count_word_by_label_topic_c[:, :, w] + beta) \
                          / (n_word_by_label_topic_c + n_vocabs * beta)
    phi[:, n_topics_c:] = (count_word_by_domain_label_topic_s[m, :, :, w] + beta) \
                          / (n_word_by_domain_label_topic_s[m] + n_vocabs * beta)
    return phi


@jit(nopython=True)
def _calc_theta(d, n_labels, n_topics_c, n_topics_s, alpha,
                n_word_by_doc_label_type, n_word_by_doc_label_topic_c, n_word_by_doc_label_topic_s):
    theta = np.zeros((n_labels, n_topics_c + n_topics_s))
    for c in range(n_topics_c):
        theta[:, c] = (n_word_by_doc_label_topic_c[d, :, c] + alpha) \
                      / (n_word_by_doc_label_type[d, :, 0] + n_topics_c * alpha)
    for s in range(n_topics_s):
        theta[:, n_topics_c + s] = (n_word_by_doc_label_topic_s[d, :, s] + alpha) \
                                   / (n_word_by_doc_label_type[d, :, 1] + n_topics_s * alpha)
    return theta


@jit(nopython=True)
def _calc_sigma(d, gamma, count_label_by_doc, n_word_by_doc_label_type):
    return (n_word_by_doc_label_type[d] + gamma) / (count_label_by_doc[d:d + 1].T + gamma.sum())


@jit(nopython=True)
def _calc_pi(d, m, n_labels, eta, n_word_by_doc, count_label_by_doc):
    if m == 0:
        return count_label_by_doc[d] / n_word_by_doc[d]
    else:
        return (count_label_by_doc[d] + eta) / (n_word_by_doc[d] + n_labels * eta)


@jit(nopython=True)
def _calc_perplexity(doc_ids, count_word_by_doc, phi_c, phi_s, theta_c, theta_s):
    sum_logp = sum_len = 0.0
    for d in doc_ids:
        sum_len += count_word_by_doc[d].sum()
        logp = 0.0
        for w in range(count_word_by_doc.shape[1]):
            if count_word_by_doc[d, w] > 0:
                s = np.sum(theta_c[d, :, :] * phi_c[:, :, w]) \
                    + np.sum(theta_s[d, :, :] * phi_s[1, :, :, w])
                if s > 0:
                    logp += count_word_by_doc[d, w] * np.log(s)
        sum_logp += logp
    return np.exp(-sum_logp / sum_len)
