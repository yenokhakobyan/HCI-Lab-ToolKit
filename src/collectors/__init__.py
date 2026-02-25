# Data collectors for different devices

try:
    from .tobii_collector import TobiiCollector
except ImportError:
    TobiiCollector = None

from .aoi_collector import AOICollector, AOIElement, AOICollection, collect_webpage_aois
from .web_hci_collector import WebHCICollectorServer

__all__ = [
    "TobiiCollector",
    "AOICollector",
    "AOIElement",
    "AOICollection",
    "collect_webpage_aois",
    "WebHCICollectorServer",
]
