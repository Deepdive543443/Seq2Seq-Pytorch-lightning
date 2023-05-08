import os, shutil
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, root = 'tensorboard'):
        '''
        Tensorboard warpper

        :param root:
        '''
        #Logger
        self.root = root
        self.global_step = 0
        self.logger = {}

        #Tensorboard
        shutil.rmtree(root)
        os.makedirs(root)
        os.makedirs(os.path.join(root, 'imgs'))
        self.writer = SummaryWriter(root)

    def update(self, key, value):
        try:
            self.logger[key]['step'] += 1
            self.logger[key]['value'] = value
        except:
            self.logger[key] = {'value': value, 'step': 0}

    def step(self):
        self.global_step += 1
        for k, v in self.logger.items():
            self.writer.add_scalar(k, v['value'], global_step=v['step'])