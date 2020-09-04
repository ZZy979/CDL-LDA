import logging

import numpy as np

from corpora import Corpus
from model.cdllda import CdlLdaModel


logger = logging.getLogger(__name__)


def run_cdllda(name, iterations=50, update_every=10,
               n_topics_c=6, n_topics_s=6, alpha=10.0, beta=0.1,
               gamma_c=1000, gamma_s=1000, eta=0.01, seed=45):
    """训练CDL-LDA模型并预测标签。"""
    corpus = Corpus(name)
    model = CdlLdaModel(
        corpus, iterations, update_every,
        n_topics_c, n_topics_s, alpha, beta, gamma_c, gamma_s, eta, seed
    )
    model.train()
    true_label = np.array([doc.label for doc in corpus if doc.domain == 1])
    pred_label = model.predict_label()
    accuracy = np.sum(pred_label == true_label) / true_label.size
    perplexity = model.calc_perplexity()
    logger.info('accuracy = {:.2%}, perplexity = {:.2f}'.format(accuracy, perplexity))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    run_cdllda('1.comVSrec', iterations=50, update_every=10, seed=45)