import logging

import numpy as np

from corpora import PriorCorpus
from model.cdllda_ex import CdlLdaExModel

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    datasets = [
        '1.comVSrec', '2.compVSsci', '3.compVStalk', '4.recVSsci', '5.recVStalk', '6.sciVStalk',
        '7.OrgsPeople', '8.OrgsPlaces', '9.PeoplePlaces',
        '10.comp-sciVSrec-talk', '11.comp-recVSsci-talk', '12.comp-talkVSrec-sci'
    ]
    for name in datasets:
        corpus = PriorCorpus(name, r'D:\IntelliJ IDEA\projects\CDLLDA\dataset')
        model = CdlLdaExModel(
            corpus, iterations=40, update_every=8, n_topics_c=6, n_topics_s_src=4, n_topics_s_tgt=6,
            alpha=10.0, beta=0.1, gamma_c=1000, gamma_s=1000, eta=0.01, use_soft=True
        )
        model.train()
        true_label = np.array([doc.label for doc in corpus if doc.domain == 1])
        pred_label = model.predict_label()
        accuracy = np.sum(pred_label == true_label) / true_label.size
        perplexity = model.calc_perplexity()
        logger.info('accuracy = {:.2%}, perplexity = {:.2f}'.format(accuracy, perplexity))
