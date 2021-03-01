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
        # self.hidden_states = None

    # TODO adapt this to the model
    def calculate_candidate_representations(self, spans, features, hidden):
        '''
        Given the candidate spans and the passages, extracts the candidates,
        calculates the condensed vector representation r_c, forms a
        correlation matrix between the candidates and calculates
        a fused representation tilda_r_Cs to represent how each candidate is
        affected by each other candidate.
        '''
        self.S_p = hidden
        self.spans = spans
        self.features = features
        self.M = np.asarray(spans[0]).shape[0] * self.k  # num_passages * num_candidates
        self.S_Cs, self.r_Cs = self.calculate_condensed_vector_representation()
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
        print("Start ind:", start_indices, "\n End ind:", end_indices,"\n")

        for p in range(self.S_p[0].shape[0]):
            # Iterate through the candidates per passage
            for i in range(self.k):
                # Start and end tokens of candidate
                sp_cb = torch.tensor(np.asarray([start_indices[p]])) # Candidate Nr. i start
                sp_ce = torch.tensor(np.asarray([end_indices[p]]))  # Candidate Nr. i end
                #print("Sp_Cb:", sp_cb, "\n Sp_Ce:", sp_ce, "\n")
                '''
                Full dimensional candidate
                Pad candidate to full length, but keep position relative to full passage
                Example p=[a,b,c,d], c=[b,c] => S_C=[0,b,c,0]
                '''
                c = (self.S_p[p][start_indices[p]:end_indices[p]])
                c_len = c.shape[0]
                num_start_pads = start_indices[p]
                num_end_pads = 256 - num_start_pads - c_len
                S_C = F.pad(input=c, pad=(0, 0, num_start_pads, num_end_pads), mode='constant', value=0)
                S_Cs.append(S_C)
                print("sp_cb shape", sp_cb.shape[0])
                print("sp_cb type ", sp_cb.type)
                print("sp_cb zero elem", sp_cb[0])
                print("wb_sp_cb", self.wb(sp_cb))
                print("sp_ce", sp_ce.shape)
                print("wb_sp_ce", self.wb(sp_ce))


                # Condensed Vector Representation
                r_C = torch.add(self.wb(sp_cb), self.we(sp_ce)).tanh()
                r_Cs.append(r_C)
                # Candidate in encoded form (embedding indices)
                enc_c = self.S_p[p][start_indices[p][i]:end_indices[p][i] + 1]
                pad_enc_c = F.pad(input=enc_c, pad=(0, 256 - c_len), mode='constant', value=0)
                encoded_candidates.append(pad_enc_c)


                # spans = torch.tensor((), dtype=torch.int64)
                # spans.new_zeros((256, 1)) #maybe use torch.tensor new_zeros
                # spans = ans_span
                # S_C = torch.stack([spans])
                # S_Cs = torch.stack([S_Cs,S_C])

                # Condensed Vector Representation

                print("Added: ",torch.add(self.wb(sp_cb), self.we(sp_ce)))
                r_C = (torch.add(self.wb(sp_cb), self.we(sp_ce))).tanh()
                print("r_C: ", r_C)
                # Try to trace in the hidden states/ encoded passages bzw sequence output
                # and put those into the linear layers wb_sp previously we used the index and not the 768
                #
                r_Cs.append(r_C)
                # Candidate in encoded form (embedding indices)
                #enc_c = self.passages[p][start_indices[p][i]:end_indices[p][i] + 1]
                #pad_enc_c = F.pad(input=enc_c, pad=(0, max_seq_len - c_len), mode='constant', value=0)
                #encoded_candidates.append(pad_enc_c)

        # Stack to turn into tensor
        S_Cs = torch.stack(S_Cs, dim=0)
        r_Cs = torch.stack(r_Cs, dim=0)
        #encoded_candidates = torch.stack(encoded_candidates, dim=0)
        return S_Cs, r_Cs #,encoded_candidates

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

