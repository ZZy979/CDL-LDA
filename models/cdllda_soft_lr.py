import logging

from models.cdllda_lr import CdlLdaLRModel
from models.cdllda_soft import CdlLdaSoftModel

logger = logging.getLogger(__name__)


class CdlLdaSoftLRModel(CdlLdaSoftModel, CdlLdaLRModel):
    """使用soft prior+逻辑回归的CDL-LDA模型"""

    name = 'CDL-LDA-soft-LR'
