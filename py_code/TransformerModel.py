# 1. The shape of the tensor passed across all modules are kept as (batch, d, seq_len)
# 2. Each module's (except for attention) output is normalized, so no need to normalize input

import torch
import numpy as np
import torch.nn as nn
from Encoder import TransformerEncoder
from Decoder import TransformerDecoder
from OutputBlock import TransformerOutput

# class Object(object):
#     pass
# self = Object()

# self = model
class Transformer(nn.Module):
    def __init__(self, model_settings, output_structure = "Vanilla"):
        super().__init__()
        torch.manual_seed(321)
        self.model_settings, self.output_structure = model_settings, output_structure
        self.encoder = TransformerEncoder(model_settings)
        self.decoder = TransformerDecoder(model_settings)
        self.linear = nn.Linear(model_settings["num_hiddens"], len(model_settings["check_loss"]) + 1)
        self.sigmoid = nn.Sigmoid()
        
        self.alpha_org, self.beta_org = model_settings["max_X"] - model_settings["min_X"], model_settings["min_X"]
        if output_structure != "Vanilla":
            model_settings2 = model_settings.copy()
            model_settings2["num_hiddens"] = model_settings["f_in"][0]
            model_settings2["num_q"] = model_settings2["num_k"] = model_settings2["num_v"] = model_settings["num_q_SAND"]
            model_settings2["dropout"] = model_settings["dropout_SAND"]
            model_settings2["check_loss"] = ()
            self.outblock = TransformerOutput(model_settings2, output_structure)
            
    def forward(self, x, y_t, emb_m_mh, e_m_mh, d_m_mh, d_m, iteration = 0, TAs_position = None, isTraining = True):
        # x, y_t, emb_m_mh, e_m_mh, d_m_mh, d_m = e_X, d_T, e_m_mh, e_m_mh, d_m_mh, d_m
        # x, y_t, emb_m_mh, e_m_mh, d_m_mh, d_m = x, d_T_full, emb_m_mh, emb_m_mh, None, None
        min_X, max_X, denoise_method = self.model_settings["min_X"], self.model_settings["max_X"], self.model_settings["denoise_method"]
        e_output = self.encoder(x, e_m_mh, emb_m_mh)
        d_output = self.decoder(y_t, e_output, e_m_mh, d_m_mh)
        org = self.linear(d_output)
        
        if self.output_structure == "Vanilla":
            return self.sigmoid(org) * self.alpha_org + self.beta_org
        else:
            # iteration = 0; TAs_position = None; isTraining = True
            org_detach = org.detach().clone()[:, :, 0].unsqueeze(-1)
            smooth = self.outblock(org_detach, y_t, d_m, iteration, TAs_position, isTraining)
            return [smooth, self.sigmoid(org) * self.alpha_org + self.beta_org]


