import numpy as np
from numba import jit
from sklearn.linear_model import LogisticRegression

from model.cdllda import CdlLdaModel, Domain, TopicType


class CdlLdaUnModel(CdlLdaModel):
    """无监督版本的CDL-LDA模型"""
    name = 'CDL-LDA-un'

    def init_word(self, doc):
        l = np.random.randint(self.corpus.n_labels)
        r = TopicType.COMMON if np.random.random() < 0.5 else TopicType.SPECIFIC
        z = np.random.randint(self.n_topics_c) if r == TopicType.COMMON \
            else np.random.randint(self.n_topics_s)
        return l, r, z

    def calc_pi(self, d, m):
        return _calc_pi(
            d, self.corpus.n_labels, self.eta, self.n_word_by_doc, self.count_label_by_doc
        )

    def predict_label(self):
        period = self.n_topics_c + self.n_topics_s
        X = np.zeros((len(self.corpus), self.corpus.n_labels * period))
        for d, doc in enumerate(self.corpus):
            for i in range(len(doc)):
                l = self.label[d][i]
                r = self.topic_type[d][i]
                z = self.topic[d][i]
                index = l * period + z if r == TopicType.COMMON \
                    else l * period + self.n_topics_c + z
                X[d][index] += 1
        X /= X.sum(axis=1)[:, np.newaxis]
        y = np.array([doc.label for doc in self.corpus])
        train_size = sum(1 for doc in self.corpus if doc.domain == Domain.SOURCE)

        # 无监督预测：使用逻辑回归
        clf = LogisticRegression(penalty='l2', tol=1e-2, C=1.0)
        clf.fit(X[:train_size], y[:train_size])
        return clf.predict(X[train_size:])


@jit(nopython=True)
def _calc_pi(d, n_labels, eta, n_word_by_doc, count_label_by_doc):
    return (count_label_by_doc[d] + eta) / (n_word_by_doc[d] + n_labels * eta)
