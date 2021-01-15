from utils.candidate_representation import Candidate_Representation
import torch.nn as nn


class Candid_rep(Candidate_Representation):

    def __init__(self, k=2):
        super(Candid_rep, self).__init__()
        self.k = k
        self.wb = nn.Linear(768, 768, bias=False)
        self.we = nn.Linear(768, 768, bias=False)
        # Linear transformations to capture the intensity
        # of each interaction (used for the attention mechanism)
        self.wc = nn.Linear(768, 768, bias=False)
        self.wo = nn.Linear(768, 768, bias=False)
        self.wv = nn.Linear(768, 1, bias=False)
