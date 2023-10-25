import re
import pprint
import logging

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ConsoleLogger(LightningLoggerBase):
    def __init__(self, log_keys=[]):
        super().__init__()
        self.log_keys = [re.compile(k) for k in log_keys]
        self.dict_printer = pprint.PrettyPrinter(indent=2, compact=False).pformat
    
    def match_log_keys(self, s):
        return True if not self.log_keys else any(r.search(s) for r in self.log_keys)

    @property
    def name(self):
        return 'console'
    
    @property
    def version(self):
        return '0'
    
    @property
    @rank_zero_experiment
    def experiment(self):
        return logging.getLogger('pytorch_lightning')
    
    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics_ = {k: v for k, v in metrics.items() if self.match_log_keys(k)}
        if not metrics_:
            return
        self.experiment.info(f"\nEpoch{metrics['epoch']} Step{step}\n{self.dict_printer(metrics_)}")

