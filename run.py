import argparse
import logging

import numpy as np

from corpora import Corpus, PriorCorpus
from model.cdllda import CdlLdaModel
from model.cdllda_ex import CdlLdaExModel
from model.cdllda_un import CdlLdaUnModel

logger = logging.getLogger(__name__)
datasets = [
    '1.comVSrec', '2.compVSsci', '3.compVStalk', '4.recVSsci', '5.recVStalk', '6.sciVStalk',
    '7.OrgsPeople', '8.OrgsPlaces', '9.PeoplePlaces',
    '10.comp-sciVSrec-talk', '11.comp-recVSsci-talk', '12.comp-talkVSrec-sci'
]


def main():
    parser = argparse.ArgumentParser(description='运行CDL-LDA模型')
    parser.add_argument(
        '-m', '--model', choices=['cdllda', 'un', 'ex'], default='cdllda', help='要使用的模型'
    )
    parser.add_argument('-p', '--path', help='数据集所在目录', dest='base_path')
    parser.add_argument(
        '-n', '--name', action='append',
        help='数据集名称，可指定多个，all表示运行全部', dest='datasets'
    )
    parser.add_argument('-i', '--iterations', type=int, default=40, help='迭代次数')
    parser.add_argument('-u', '--update-every', type=int, default=8, help='每迭代多少次更新参数')
    parser.add_argument('-c', '--n-topics-c', type=int, default=6, help='公共主题数量')
    parser.add_argument('-s', '--n-topics-s', type=int, default=6, help='（源域）特有主题数量')
    parser.add_argument('--n-topics-s-tgt', type=int, default=6, help='目标域特有主题数量（仅用于CDL-LDA-ex）')
    parser.add_argument('-a', '--alpha', type=float, default=10.0, help='文档-主题分布θ的Dirichlet先验')
    parser.add_argument('-b', '--beta', type=float, default=0.1, help='主题-单词分布φ的Dirichlet先验')
    parser.add_argument('--gamma-c', type=float, default=1000, help='公共主题类型分布σ的Beta先验')
    parser.add_argument('--gamma-s', type=float, default=1000, help='特有主题类型分布σ的Beta先验')
    parser.add_argument('-e', '--eta', type=float, default=0.01, help='标签（主题组）分布π的Dirichlet先验')
    parser.add_argument('--use-soft', action='store_true', help='单词标签是否使用soft prior')

    args = parser.parse_args()
    global datasets
    if 'all' not in args.datasets:
        datasets = args.datasets
    model_cls = CdlLdaModel if args.model == 'cdllda' \
        else CdlLdaUnModel if args.model == 'un' \
        else CdlLdaExModel
    corpus_cls = PriorCorpus if args.model == 'ex' else Corpus
    model_args = {
        'iterations': args.iterations,
        'update_every': args.update_every,
        'n_topics_c': args.n_topics_c,
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma_c': args.gamma_c,
        'gamma_s': args.gamma_s,
        'eta': args.eta
    }
    if args.model == 'ex':
        model_args['n_topics_s_src'] = args.n_topics_s
        model_args['n_topics_s_tgt'] = args.n_topics_s_tgt
        model_args['use_soft'] = args.use_soft
    else:
        model_args['n_topics_s'] = args.n_topics_s
    for name in datasets:
        corpus = corpus_cls(name, args.base_path)
        model_args['corpus'] = corpus
        model = model_cls(**model_args)
        model.train()
        true_label = np.array([doc.label for doc in corpus if doc.domain == 1])
        pred_label = model.predict_label()
        accuracy = np.sum(pred_label == true_label) / true_label.size
        perplexity = model.calc_perplexity()
        logger.info('accuracy = {:.2%}, perplexity = {:.2f}'.format(accuracy, perplexity))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()
