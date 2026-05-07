from .base import *
from .arena_55k import *
from .arena_100k import *
from .lmsys_chat_1m import *
from .wildchat import *
from .wildchat_4_8 import *
from .arena_140k import *


dset_class_map: dict[str, type[ConvoDataset]] = {
    'arena_55k': Arena55kDataset,
    'arena_100k': Arena100kDataset,
    'wildchat': WildchatDataset,
    'lmsys_1m': LMSYSChat1mDataset,
    'wildchat_4_8': Wildchat48Dataset,
    'arena_140k': Arena140kDataset,
    'arena_140k_fixed': Arena140kFixedDataset
}
