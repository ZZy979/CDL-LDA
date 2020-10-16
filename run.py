import argparse
import logging
import os
import pickle

import numpy as np

from corpora import CrossDomainCorpus
from models import *

logger = logging.getLogger(__name__)
MODELS = {
    'cdllda': CdlLdaModel,
    'un': CdlLdaUnModel,
    'soft': CdlLdaSoftModel,
    'lr': CdlLdaLRModel,
    'soft-lr': CdlLdaSoftLRModel
}
DATASETS = [
    '1.comVSrec', '2.compVSsci', '3.compVStalk', '4.recVSsci', '5.recVStalk', '6.sciVStalk',
    '7.OrgsPeople', '8.OrgsPlaces', '9.PeoplePlaces',
    '10.comp-sciVSrec-talk', '11.comp-recVSsci-talk', '12.comp-talkVSrec-sci'
]


def main():
    parser = argparse.ArgumentParser(description='运行CDL-LDA模型')
    parser.add_argument(
        '-m', '--model', choices=['cdllda', 'un', 'soft', 'lr', 'soft-lr'],
        default='cdllda', help='要使用的模型'
    )
    parser.add_argument('-p', '--path', help='数据集所在目录', dest='base_path')
    parser.add_argument(
        '-n', '--name', action='append',
        help='数据集名称，可指定多个，all表示运行全部', dest='datasets'
    )
    parser.add_argument('-i', '--iterations', type=int, default=40, help='迭代次数')
    parser.add_argument('-u', '--update-every', type=int, default=8, help='每迭代多少次更新参数')
    parser.add_argument(
        '--gibbs-iter', type=int, default=4,
        help='吉布斯采样迭代次数（仅用于CDL-LDA-LR和CDL-LDA-soft-LR）'
    )
    parser.add_argument('-g', '--n-groups', type=int, default=2, help='主题组数量（仅用于CDL-LDA-LR）')
    parser.add_argument('-c', '--n-topics-c', type=int, default=6, help='公共主题数量')
    parser.add_argument('-s', '--n-topics-s', type=int, default=6, help='特有主题数量')
    parser.add_argument('-a', '--alpha', type=float, default=10.0, help='文档-主题分布θ的Dirichlet先验')
    parser.add_argument('-b', '--beta', type=float, default=0.01, help='主题-单词分布φ的Dirichlet先验')
    parser.add_argument('--gamma-c', type=float, default=1000, help='公共主题类型分布σ的Beta先验')
    parser.add_argument('--gamma-s', type=float, default=1000, help='特有主题类型分布σ的Beta先验')
    parser.add_argument('-e', '--eta', type=float, default=0.1, help='主题组分布π的Dirichlet先验')
    parser.add_argument(
        '--use-soft', action='store_true',
        help='是否使用soft prior（仅用于CDL-LDA-soft）和CDL-LDA-soft-LR'
    )

    args = parser.parse_args()
    datasets = DATASETS if 'all' in args.datasets else args.datasets
    model_cls = MODELS[args.model]
    model_args = {
        'iterations': args.iterations,
        'update_every': args.update_every,
        'n_topics_c': args.n_topics_c,
        'n_topics_s': args.n_topics_s,
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma_c': args.gamma_c,
        'gamma_s': args.gamma_s,
        'eta': args.eta
    }
    if args.model in ('soft', 'soft-lr'):
        model_args['use_soft'] = args.use_soft
    if args.model in ('lr', 'soft-lr'):
        model_args['gibbs_iter'] = args.gibbs_iter
        if args.model == 'lr':
            # CDL-LDA-Soft-LR的主题组个数由prior的维数决定
            model_args['n_groups'] = args.n_groups

    for name in datasets:
        with open(os.path.join(args.base_path, name, 'data.pkl'), 'rb') as f:
            docs = pickle.load(f)
            word2id = pickle.load(f)
            id2word = pickle.load(f)
            labels = pickle.load(f)
            domains = pickle.load(f)
        corpus = CrossDomainCorpus(name, docs, labels, domains)
        model_args['corpus'] = corpus
        model_args['id2word'] = id2word
        if args.model in ('soft', 'soft-lr') and args.use_soft:
            with open(os.path.join(args.base_path, name, 'soft_prior.pkl'), 'rb') as f:
                model_args['prior'] = np.array(pickle.load(f))
            if args.model == 'soft-lr':
                model_args['n_groups'] = model_args['prior'].shape[1]

        model = model_cls(**model_args)
        model.estimate()
        true_label = np.array([doc.label for doc in corpus if doc.domain == 1])
        pred_label = model.predict_label()
        accuracy = np.sum(pred_label == true_label) / true_label.size
        perplexity = model.calc_perplexity()
        logger.info('accuracy = {:.2%}, perplexity = {:.2f}'.format(accuracy, perplexity))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()
