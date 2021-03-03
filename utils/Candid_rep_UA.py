import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Candid_rep():

    def __init__(self, k=1):
        super(Candid_rep, self).__init__()
        self.k = k
        self.wb = nn.Linear(768, 256, bias=False)
        self.we = nn.Linear(768, 256, bias=False)
        # Linear transformations to capture the intensity
        # of each interaction (used for the attention mechanism)
        self.wc = nn.Linear(256, 256, bias=False)
        self.wo = nn.Linear(256, 256, bias=False)
        self.wv = nn.Linear(256, 1, bias=False)


    # TODO adapt this to the model
    def calculate_candidate_representations(self, spans, features, seq_outpu):
        '''
        Given the candidate spans and the passages, extracts the candidates,
        calculates the condensed vector representation r_c, forms a
        correlation matrix between the candidates and calculates
        a fused representation tilda_r_Cs to represent how each candidate is
        affected by each other candidate.
        '''
        self.S_p = seq_outpu
        self.spans = spans
        self.features = features
        self.M = np.asarray(spans[0]).shape[0] * self.k  # num_passages * num_candidates
        self.S_Cs, self.r_Cs, self.encoded_candidates = self.calculate_condensed_vector_representation()
        self.V = self.calculate_correlations()
        self.tilda_r_Cs = self.generate_fused_representation()

    def calculate_condensed_vector_representation(self):
        '''
        Returns the condensed vector representation of all candidates by condensing their start and end tokens.
        :return:
        '''
        S_Cs = []
        r_Cs = []
        encoded_candidates = []
        start_indices = self.spans[0]
        end_indices = self.spans[1]
        print("Start ind:", start_indices, "\n End ind:", end_indices, "\n")
        #print("Sequence_Output", self.S_p, "Shape ",self.S_p.shape, "\n")
        #print("Sequence_Output_len", len(self.S_p), "Shape ", self.S_p.shape[0], "\n")
        #print("S_P.shape", self.S_p.shape[0])
        for p in range(self.S_p.shape[0]):
            # Iterate through the candidates per passage

            # Start and end tokens of candidate
            sp_cb = self.S_p[p][start_indices[p]]  # Candidate Nr. i start
            sp_ce = self.S_p[p][end_indices[p]]  # Candidate Nr. i end

            '''
            Full dimensional candidate
            Pad candidate to full length, but keep position relative to full passage
            Example p=[a,b,c,d], c=[b,c] => S_C=[0,b,c,0]
            '''

            c = (self.S_p[p][start_indices[p]:end_indices[p]]) #position of the answer with shape[...,768]
            c_len = c.shape[0]

            num_start_pads = start_indices[p]
            num_end_pads = 256 - num_start_pads - c_len
            S_C = F.pad(input=c, pad=(0, 0, num_start_pads, num_end_pads), mode='constant', value=0)
            S_Cs.append(S_C)

            # Condensed Vector Representation
            r_C = torch.add(self.wb(sp_cb), self.we(sp_ce)).tanh()
            r_Cs.append(r_C)
            # Candidate in encoded form (embedding indices)
            enc_c = self.S_p[p][start_indices[p]:end_indices[p] + 1]
            pad_enc_c = F.pad(input=enc_c, pad=(0, 256 - c_len), mode='constant', value=0)
            encoded_candidates.append(pad_enc_c)

        # Stack to turn into tensor
        S_Cs = torch.stack(S_Cs, dim=0)
        r_Cs = torch.stack(r_Cs, dim=0)
        encoded_candidates = torch.stack(encoded_candidates, dim=0)

        return S_Cs, r_Cs, encoded_candidates

    def calculate_correlations(self):
        '''
        Model the interactions via attention mechanism.
        Returns a correlation matrix
        '''

        V_jms = []

        for i, r_C in enumerate(self.r_Cs):
            rcm = torch.cat([self.r_Cs[0:i], self.r_Cs[i + 1:]], dim=0)
            c = self.wc(r_C)
            o = self.wo(rcm)
            V_jm = self.wv(torch.add(c, o).tanh())
            V_jms.append(V_jm)

        V = torch.stack(V_jms, dim=0)

        V = V.view(V.shape[0], V.shape[1])
        return V

    def generate_fused_representation(self):
        # Normalize interactions

        alpha_ms = []
        for i, V_jm in enumerate(self.V):
            numerator = torch.exp(V_jm)
            denominator_correlations = torch.cat([self.V[0:i], self.V[i + 1:]], dim=0)  # (200x199)

            denominator = torch.sum(torch.exp(denominator_correlations), dim=0)
            alpha_m = torch.div(numerator, denominator)  # 199x1
            alpha_ms.append(alpha_m)
        alpha = torch.stack(alpha_ms, dim=0)  # (200,199)
        tilda_rcms = []
        for i, r_C in enumerate(self.r_Cs):
            rcm = torch.cat([self.r_Cs[0:i], self.r_Cs[i + 1:]], dim=0)
            alpha_m = torch.cat([alpha[0:i], alpha[i + 1:]], dim=0)  # (199x199)
            tilda_rcm = torch.sum(torch.mm(alpha_m, rcm), dim=0)  # (1x100)
            tilda_rcms.append(tilda_rcm)

        return torch.stack(tilda_rcms, dim=0)  # (200x1x100)

