from .__version__ import version
from .scripts import _utils, __credentials__, critters, \
    keyword_stats, knapsack, locksmith

name='ADvantage'
# thisdir = os.path.split(os.path.realpath(__file__))[0]
# itlist = os.listdir(thisdir)
# __all__ = [os.path.split(x)[-1].strip('.py') for x in itlist if x.endswith(
#     '.py') and not x.endswith('__init__.py')]

__version__ = version
__all__ = [
    "__version__",
    "_utils",
    "__credentials__.py",
    "critters",
    "keyword_stats",
    "knapsack",
    "locksmith",
    "resources"
]

