import logging

import numpy as np

from corpora import Corpus
from model.cdllda_un import CdlLdaUnModel

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    corpus = Corpus('1.comVSrec', r'D:\IntelliJ IDEA\projects\CDLLDA\dataset')
    model = CdlLdaUnModel(
        corpus, iterations=40, update_every=8, n_topics_c=6, n_topics_s=6,
        alpha=10.0, beta=0.1, gamma_c=1000, gamma_s=1000, eta=0.01
    )
    model.train()
    true_label = np.array([doc.label for doc in corpus if doc.domain == 1])
    pred_label = model.predict_label()
    accuracy = np.sum(pred_label == true_label) / true_label.size
    perplexity = model.calc_perplexity()
    logger.info('accuracy = {:.2%}, perplexity = {:.2f}'.format(accuracy, perplexity))
