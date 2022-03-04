# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on the HeR implementation at https://github.com/TianhongDai/hindsight-experience-replay.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# from the original repo. needed for replaying original experts to collect demos.
class actor_orig(nn.Module):
    def __init__(self, env_params):
        super(actor_orig, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        n_obj = env_params['n_objects']
        inp_dim = env_params['gripper'] + n_obj * (env_params['object'] + env_params['goal'])
        self.fc1 = nn.Linear(inp_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, grip, obj, g):
        batch_dims = obj.shape[:-2]
        obj = obj.reshape(*batch_dims, -1)
        g = g.reshape(*batch_dims, -1)
        x = torch.cat((grip, obj, g), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        n_obj = env_params['n_objects']
        inp_dim = env_params['gripper'] + n_obj * (env_params['object'] + env_params['goal']) + env_params['action']
        self.fc1 = nn.Linear(inp_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, grip, obj, g, actions):
        batch_dims = obj.shape[:-2]
        obj = obj.reshape(*batch_dims, -1)
        g = g.reshape(*batch_dims, -1)
        x = torch.cat((grip, obj, g, actions / self.max_action), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class EncoderLayer(nn.Module):
    """Adapted from: https://github.com/jwang0306/transformer-pytorch."""
    def __init__(self, hidden_size, n_head, dim_ff, pre_lnorm, dropout=0.0):
        super(EncoderLayer, self).__init__()
        # self-attention part
        self.self_attn = nn.MultiheadAttention(hidden_size, n_head, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(hidden_size)

        # feed forward network part
        self.pff = nn.Sequential(
            nn.Linear(hidden_size, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, hidden_size),
            nn.Dropout(dropout)
        )
        self.pff_norm = nn.LayerNorm(hidden_size)
        self.pre_lnorm = pre_lnorm

    def forward(self, src, src_mask=None):
        if self.pre_lnorm:
            pre = self.self_attn_norm(src)
            # residual connection
            src = src + self.dropout(self.self_attn(pre, pre, pre, src_mask)[0])
            pre = self.pff_norm(src)
            src = src + self.pff(pre) # residual connection
        else:
            # residual connection + layerNorm
            src2 = self.dropout(self.self_attn(src, src, src, src_mask)[0])
            src = self.self_attn_norm(src + src2)
            src = self.pff_norm(src + self.pff(src)) # residual connection + layerNorm
        return src


# define the actor network
class actor_tfm(nn.Module):
    def __init__(self, env_params, pos_enc=False, embed_dim=256, dim_ff=256, n_head=4, n_blocks=2, dropout_p=0.0):
        super(actor_tfm, self).__init__()
        self.max_action = env_params['action_max']
        token_dim = 0
        for key in ['gripper', 'object', 'goal']:
            token_dim += env_params[key]
        self.embed = nn.Linear(token_dim, embed_dim)
        self.enc = nn.Sequential(*[EncoderLayer(embed_dim, n_head=n_head, dim_ff=dim_ff, pre_lnorm=True, dropout=dropout_p) for _ in range(n_blocks)])
        self.action_out = nn.Linear(embed_dim, env_params['action'])
        self.pos_enc = None
        if pos_enc:
            self.pos_enc = PositionalEncoding(embed_dim)

    def forward(self, grip, obj, g):
        grip = torch.unsqueeze(grip, -2).expand(-1, obj.shape[-2], -1)
        x = torch.cat((grip, obj, g), -1)
        x = self.embed(x).transpose(0, 1)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.enc(x)
        x = self.action_out(x.transpose(0, 1)).sum(-2)
        return self.max_action * torch.tanh(x)


class critic_tfm(nn.Module):
    def __init__(self, env_params, pos_enc=False, embed_dim=256, dim_ff=256, n_head=4, n_blocks=2, dropout_p=0.0):
        super(critic_tfm, self).__init__()
        self.max_action = env_params['action_max']
        token_dim = env_params['action']
        for key in ['gripper', 'object', 'goal']:
            token_dim += env_params[key]
        self.embed = nn.Linear(token_dim, embed_dim)
        self.enc = nn.Sequential(*[EncoderLayer(embed_dim, n_head=n_head, dim_ff=dim_ff, pre_lnorm=True, dropout=dropout_p) for _ in range(n_blocks)])
        self.q_out = nn.Linear(embed_dim, 1)
        self.pos_enc = None
        if pos_enc:
            self.pos_enc = PositionalEncoding(embed_dim)

    def forward(self, grip, obj, g, actions):
        grip = torch.unsqueeze(grip, -2).expand(-1, obj.shape[-2], -1)
        actions = torch.unsqueeze(actions, -2).expand(-1, obj.shape[-2], -1)
        x = torch.cat((grip, obj, g, actions / self.max_action), -1)
        x = self.embed(x).transpose(0, 1)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.enc(x)
        return self.q_out(x.transpose(0, 1)).sum(-2)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class actor_deepset(nn.Module):
    def __init__(self, env_params, agg='mean'):
        super(actor_deepset, self).__init__()
        self.max_action = env_params['action_max']
        token_dim = 0
        for key in ['gripper', 'object', 'goal']:
            token_dim += env_params[key]
        self.fc1 = nn.Linear(token_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        self.agg = agg

    def forward(self, grip, obj, g):
        extra_dims = (-1,) * (grip.ndim - 1)
        grip = torch.unsqueeze(grip, -2).expand(*extra_dims, obj.shape[-2], -1)
        x = torch.cat((grip, obj, g), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.agg == "sum":
            x = x.sum(-2)
        elif self.agg == "mean":
            x = x.mean(-2)
        elif self.agg == "max":
            x = x.max(-2).values
        else:
            raise ValueError(f"Unrecognized aggregation function {self.agg}")
        x = self.action_out(x)
        return self.max_action * torch.tanh(x)


class critic_deepset(nn.Module):
    def __init__(self, env_params, agg='mean'):
        super(critic_deepset, self).__init__()
        self.max_action = env_params['action_max']
        token_dim = env_params['action']
        for key in ['gripper', 'object', 'goal']:
            token_dim += env_params[key]
        self.fc1 = nn.Linear(token_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.agg = agg

    def forward(self, grip, obj, g, actions):
        extra_dims = (-1,) * (grip.ndim - 1)
        grip = torch.unsqueeze(grip, -2).expand(*extra_dims, obj.shape[-2], -1)
        actions = torch.unsqueeze(actions, -2).expand(-1, obj.shape[-2], -1)
        x = torch.cat((grip, obj, g, actions / self.max_action), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.agg == "sum":
            x = x.sum(-2)
        elif self.agg == "mean":
            x = x.mean(-2)
        elif self.agg == "max":
            x = x.max(-2).values
        else:
            raise ValueError(f"Unrecognized aggregation function {self.agg}")
        x = F.relu(self.fc3(x))
        x = self.q_out(x)
        return x


# define the actor network
class actor_padded(nn.Module):
    tot_objs = 6
    def __init__(self, env_params):
        super(actor_padded, self).__init__()
        self.max_action = env_params['action_max']
        inp_dim = env_params['gripper'] + self.tot_objs * (env_params['object'] + env_params['goal'])
        self.fc1 = nn.Linear(inp_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, grip, obj, g):
        batch_dims = obj.shape[:-2]
        n_obj = obj.shape[-2]
        if n_obj < self.tot_objs:
            obj_padding = torch.zeros(*batch_dims, self.tot_objs - n_obj, obj.shape[-1]).to(obj.device)
            obj = torch.cat((obj, obj_padding), dim=-2)
            g_padding = torch.zeros(*batch_dims, self.tot_objs - n_obj, g.shape[-1]).to(obj.device)
            g = torch.cat((g, g_padding), dim=-2)
        obj = obj.reshape(*batch_dims, -1)
        g = g.reshape(*batch_dims, -1)
        x = torch.cat((grip, obj, g), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions


class critic_padded(nn.Module):
    tot_objs = 6
    def __init__(self, env_params):
        super(critic_padded, self).__init__()
        self.max_action = env_params['action_max']
        inp_dim = env_params['gripper'] + 6 * (env_params['object'] + env_params['goal']) + env_params['action']
        self.fc1 = nn.Linear(inp_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, grip, obj, g, actions):
        batch_dims = obj.shape[:-2]
        n_obj = obj.shape[-2]
        if n_obj < self.tot_objs:
            obj_padding = torch.zeros(*batch_dims, self.tot_objs - n_obj, obj.shape[-1]).to(obj.device)
            obj = torch.cat((obj, obj_padding), dim=-2)
            g_padding = torch.zeros(*batch_dims, self.tot_objs - n_obj, g.shape[-1]).to(obj.device)
            g = torch.cat((g, g_padding), dim=-2)
        obj = obj.reshape(*batch_dims, -1)
        g = g.reshape(*batch_dims, -1)
        x = torch.cat((grip, obj, g, actions / self.max_action), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class actor_deepset_big(nn.Module):
    def __init__(self, env_params, agg='mean'):
        super(actor_deepset_big, self).__init__()
        self.max_action = env_params['action_max']
        token_dim = 0
        for key in ['gripper', 'object', 'goal']:
            token_dim += env_params[key]
        self.fc1 = nn.Linear(token_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        self.agg = agg

    def forward(self, grip, obj, g):
        extra_dims = (-1,) * (grip.ndim - 1)
        grip = torch.unsqueeze(grip, -2).expand(*extra_dims, obj.shape[-2], -1)
        x = torch.cat((grip, obj, g), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.agg == "sum":
            x = x.sum(-2)
        elif self.agg == "mean":
            x = x.mean(-2)
        elif self.agg == "max":
            x = x.max(-2).values
        else:
            raise ValueError(f"Unrecognized aggregation function {self.agg}")
        x = F.relu(self.fc4(x))
        x = self.action_out(x)
        return self.max_action * torch.tanh(x)


class critic_deepset_big(nn.Module):
    def __init__(self, env_params, agg='mean'):
        super(critic_deepset_big, self).__init__()
        self.max_action = env_params['action_max']
        token_dim = env_params['action']
        for key in ['gripper', 'object', 'goal']:
            token_dim += env_params[key]
        self.fc1 = nn.Linear(token_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.agg = agg

    def forward(self, grip, obj, g, actions):
        extra_dims = (-1,) * (grip.ndim - 1)
        grip = torch.unsqueeze(grip, -2).expand(*extra_dims, obj.shape[-2], -1)
        actions = torch.unsqueeze(actions, -2).expand(-1, obj.shape[-2], -1)
        x = torch.cat((grip, obj, g, actions / self.max_action), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.agg == "sum":
            x = x.sum(-2)
        elif self.agg == "mean":
            x = x.mean(-2)
        elif self.agg == "max":
            x = x.max(-2).values
        else:
            raise ValueError(f"Unrecognized aggregation function {self.agg}")
        x = F.relu(self.fc4(x))
        x = self.q_out(x)
        return x
