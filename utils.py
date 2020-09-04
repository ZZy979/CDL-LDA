import numpy as np
from numba import jit, guvectorize


@jit(nopython=True)
def seed(s):
    np.random.seed(s)


@guvectorize(["void(float64[:], int32[:])"], '(n)->()')
def choice(p, res):
    """按给定的概率从0~n-1中随机选择一个数并返回，p[i]表示i被选中的概率。"""
    x = np.random.rand()
    cum = 0
    for i, p in enumerate(p):
        cum += p
        if x < cum:
            res[0] = i
            break


@jit(nopython=True)
def calc_joint_distribution(
        d, m, w, n_labels, n_vocabs, n_topics_c, n_topics_s, alpha, beta, gamma, eta,
        n_word_by_doc, count_label_by_doc,
        n_word_by_doc_label_type, n_word_by_doc_label_topic_c, n_word_by_doc_label_topic_s,
        count_word_by_label_topic_c, count_word_by_domain_label_topic_s,
        n_word_by_label_topic_c, n_word_by_domain_label_topic_s
):
    # 属于域m、标签l的主题z产生单词w的概率，ndarray(L, TC + TS), 0~TC-1列表示公共主题，TC~TC+TS-1列表示特有主题
    phi = np.zeros((n_labels, n_topics_c + n_topics_s))
    phi[:, :n_topics_c] = (count_word_by_label_topic_c[:, :, w] + beta) \
                          / (n_word_by_label_topic_c + n_vocabs * beta)
    phi[:, n_topics_c:] = (count_word_by_domain_label_topic_s[m, :, :, w] + beta) \
                          / (n_word_by_domain_label_topic_s[m] + n_vocabs * beta)

    # 文档d的标签-主题联合分布，ndarray(L, TC + TS), 0~TC-1列表示公共主题，TC~TC+TS-1列表示特有主题
    theta = np.zeros((n_labels, n_topics_c + n_topics_s))
    for c in range(n_topics_c):
        theta[:, c] = (n_word_by_doc_label_topic_c[d, :, c] + alpha) \
                      / (n_word_by_doc_label_type[d, :, 0] + n_topics_c * alpha)
    for s in range(n_topics_s):
        theta[:, n_topics_c + s] = (n_word_by_doc_label_topic_s[d, :, s] + alpha) \
                                   / (n_word_by_doc_label_type[d, :, 1] + n_topics_s * alpha)

    # 文档d的标签-主题类型联合分布，ndarray(L, R)
    sigma = (n_word_by_doc_label_type[d] + gamma) / (count_label_by_doc[d:d + 1].T + gamma.sum())

    # 属于域m的文档d的标签概率分布，ndarray(L, 1)
    if m == 0:
        pi = count_label_by_doc[d:d + 1].T / n_word_by_doc[d]
    else:
        pi = (count_label_by_doc[d:d + 1].T + eta) / (n_word_by_doc[d] + n_labels * eta)

    distribution = phi * theta
    distribution[:, :n_topics_c] *= sigma[:, :1]
    distribution[:, n_topics_c:] *= sigma[:, 1:]
    distribution *= pi
    distribution /= distribution.sum()
    return distribution


@jit(nopython=True)
def calc_perplexity(doc_ids, count_word_by_doc, phi_c, phi_s, theta_c, theta_s):
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
