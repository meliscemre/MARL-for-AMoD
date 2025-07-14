import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.distributions import Dirichlet, Beta
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
import random
import json



#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, mode=0, edges=None):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        if mode == 0:
            self.lin3 = nn.Linear(hidden_size, 1)
        elif mode == 1:
            self.lin3 = nn.Linear(hidden_size, 2)
        else:
            self.lin3 = nn.Linear(hidden_size, 3)

    def forward(self, state, edge_index, deterministic=False):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1)
        if deterministic:
            if self.mode == 0:
                action = (concentration) / (concentration.sum() + 1e-20)
            elif self.mode == 1:
                action_o = (concentration[:,:,0])/(concentration[:,:,0] + concentration[:,:,1] + 1e-10)
                action_o[action_o<0] = 0
                action = action_o.squeeze(0).unsqueeze(-1)
            else:
                action_o = (concentration[:,:,0])/(concentration[:,:,0] + concentration[:,:,1] + 1e-10)
                action_o[action_o<0] = 0
                action_reb = (concentration[:,:,2]) / (concentration[:,:,2].sum() + 1e-10)
                action = torch.cat((action_o.squeeze(0).unsqueeze(-1), action_reb.squeeze(0).unsqueeze(-1)),-1)
            log_prob = None
        else:
            if self.mode == 0:
                m = Dirichlet(concentration + 1e-20)
                action = m.rsample()
                log_prob = m.log_prob(action)
                action = action.squeeze(0).unsqueeze(-1)
            elif self.mode == 1:
                m_o = Beta(concentration[:,:,0] + 1e-10, concentration[:,:,1] + 1e-10)
                action_o = m_o.rsample()
                log_prob = m_o.log_prob(action_o).sum(dim=-1)
                action = action_o.squeeze(0).unsqueeze(-1)             
            else:        
                m_o = Beta(concentration[:,:,0] + 1e-10, concentration[:,:,1] + 1e-10)
                action_o = m_o.rsample()
                # Rebalancing desired distribution
                m_reb = Dirichlet(concentration[:,:,-1] + 1e-10)
                action_reb = m_reb.rsample()              
                log_prob = m_o.log_prob(action_o).sum(dim=-1) + m_reb.log_prob(action_reb)
                action = torch.cat((action_o.squeeze(0).unsqueeze(-1), action_reb.squeeze(0).unsqueeze(-1)),-1)
       
        return action, log_prob
    

class GNNActor1(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Beta Policy. OD-based action.
    """

    def __init__(self, edges, in_channels, hidden_size=128, act_dim=10, mode=1):
        super().__init__()
        self.in_channels = in_channels
        self.nregion = act_dim
        self.hidden = hidden_size
        self.edges = edges
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(2*in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, 2)

    def forward(self, state, edge_index, deterministic=False):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.nregion, self.in_channels)

        # Obtain edge features using 'out' for the updated node features
        edges_src = self.edges[:, 0] # [E]
        edges_dst = self.edges[:, 1] # [E]

        # Obtain features for each node involved in an edge
        edge_features_src = x[:, edges_src, : ] # [#batch, E, #features]
        edge_features_dst = x[:, edges_dst, : ] # [#batch, E, #features]

        # Concatenate features from source and destination nodes
        edge_features = torch.cat([edge_features_src, edge_features_dst], dim=2)

        x = F.leaky_relu(self.lin1(edge_features))
        x = F.softplus(self.lin2(x))
        concentration = x.squeeze(-1)
        if deterministic:
            action = (concentration[:,:,0])/(concentration[:,:,0] + concentration[:,:,1] + 1e-10)
            action[action<0] = 0
            action = action.reshape(-1,self.nregion,self.nregion).squeeze(0)
            log_prob = None
        else:
            m = Beta(concentration[:,:,0] + 1e-10, concentration[:,:,1] + 1e-10)
            action = m.rsample()
            log_prob = m.log_prob(action).sum(dim=-1)
            action = action.reshape(-1,self.nregion,self.nregion).squeeze(0)
        return action, log_prob
    
class MLPActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Beta Policy. OD-based actoin.
    """

    def __init__(self, in_channels, hidden_size=128, act_dim=10, mode=1, edges=None):
        super().__init__()
        self.in_channels = in_channels
        self.nregion = act_dim
        self.lin1 = nn.Linear(in_channels*self.nregion, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, 2*self.nregion*self.nregion)

    def forward(self, state, edge_index, deterministic=False):

        x = state.reshape(-1, self.nregion * self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.leaky_relu(self.lin3(x))
        x = F.softplus(self.lin4(x))
        concentration = x.squeeze(-1)
        if deterministic:
            action = (concentration[:,:self.nregion*self.nregion])/(concentration[:,:self.nregion*self.nregion] + concentration[:,self.nregion*self.nregion:] + 1e-10)
            action[action<0] = 0
            action = action.reshape(-1,self.nregion,self.nregion).squeeze(0)
            log_prob = None
        else:
            m = Beta(concentration[:,:self.nregion*self.nregion] + 1e-10, concentration[:,self.nregion*self.nregion:] + 1e-10)
            action = m.rsample().squeeze(0)
            log_prob = m.log_prob(action).sum(dim=-1)   
            action = action.reshape(-1,self.nregion,self.nregion).squeeze(0)       
        return action, log_prob
    
class MLPActor1(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Beta Policy. Origin-based action.
    """

    def __init__(self, in_channels, hidden_size=128, act_dim=10, mode=1, edges=None):
        super().__init__()
        self.in_channels = in_channels
        self.nregion = act_dim
        self.mode = mode
        self.lin1 = nn.Linear(in_channels*self.nregion, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        if self.mode == 0:
            self.lin4 = nn.Linear(hidden_size, self.nregion)
        elif self.mode == 1:
            self.lin4 = nn.Linear(hidden_size, 2*self.nregion)
        else:
            self.lin4 = nn.Linear(hidden_size, 3*self.nregion)

    def forward(self, state, edge_index, deterministic=False):

        x = state.reshape(-1, self.nregion * self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.leaky_relu(self.lin3(x))
        x = F.softplus(self.lin4(x))
        concentration = x.squeeze(-1)
        if deterministic:
            if self.mode == 0:
                action = (concentration) / (concentration.sum() + 1e-20)
                action = action.reshape(-1,self.nregion,1)
            elif self.mode == 1:
                action = (concentration[:,:self.nregion]-1)/(concentration[:,:self.nregion] + concentration[:,self.nregion:] -2 + 1e-10)
                action[action<0] = 0
                action = action.reshape(-1,self.nregion,1)
            else:
                action_o = (concentration[:,:self.nregion]-1)/(concentration[:,:self.nregion] + concentration[:,self.nregion:2*self.nregion] -2 + 1e-10)
                action_o[action_o<0] = 0
                action_reb = (concentration[:,2*self.nregion:]) / (concentration[:,2*self.nregion:].sum() + 1e-10)
                action = torch.cat((action_o.reshape(-1,self.nregion,1).squeeze(0), action_reb.reshape(-1,self.nregion,1).squeeze(0)),-1)
            log_prob = None
        else:
            if self.mode == 0:
                m = Dirichlet(concentration + 1e-20)
                action = m.rsample()
                log_prob = m.log_prob(action)
                action = action.reshape(-1,self.nregion,1).squeeze(0)
            elif self.mode == 1:
                m = Beta(concentration[:,:self.nregion] + 1e-10, concentration[:,self.nregion:] + 1e-10)
                action = m.rsample().squeeze(0)
                log_prob = m.log_prob(action).sum(dim=-1)   
                action = action.reshape(-1,self.nregion,1).squeeze(0)                
            else:        
                m_o = Beta(concentration[:,:self.nregion] + 1e-10, concentration[:,self.nregion:2*self.nregion] + 1e-10)
                action_o = m_o.rsample().squeeze(0)
                # Rebalancing desired distribution
                m_reb = Dirichlet(concentration[:,2*self.nregion:] + 1e-10)
                action_reb = m_reb.rsample().squeeze(0)              
                log_prob = m_o.log_prob(action_o).sum(dim=-1) + m_reb.log_prob(action_reb)
                action = torch.cat((action_o.reshape(-1,self.nregion,1).squeeze(0), action_reb.reshape(-1,self.nregion,1).squeeze(0)),-1)       
        return action, log_prob

#########################################
############## CRITIC ###################
#########################################


class GNNCritic1(nn.Module):
    """
    Architecture 1, GNN, Pointwise Multiplication, Readout, FC
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6, edges=None):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        action = action * 10
        action = action.unsqueeze(-1)  # (B,N,1)
        x = x * action  # pointwise multiplication (B,N,21)
        x = x.sum(dim=1)  # (B,21)
        x = F.relu(self.lin1(x))  # (B,H)
        x = F.relu(self.lin2(x))  # (B,H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic2(nn.Module):
    """
    Architecture 2, GNN, Readout, Concatenation, FC
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6, edges=None):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + act_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, 21)  # (B,N,21)
        x = torch.sum(x, dim=1)  # (B, 21)
        concat = torch.cat([x, action], dim=-1)  # (B, 21+N)
        x = F.relu(self.lin1(concat))  # (B,H)
        x = F.relu(self.lin2(x))  # (B,H)
        x = self.lin3(x).squeeze(-1)  # B
        return x


class GNNCritic3(nn.Module):
    """
    Architecture 3: Concatenation, GNN, Readout, FC
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, edges=None):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(22, 22)
        self.lin1 = nn.Linear(22, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        cat = torch.cat([state, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        out = F.relu(self.conv1(cat, edge_index))
        x = out + cat
        x = x.reshape(-1, self.act_dim, 22)  # (B,N,22)
        x = F.relu(self.lin1(x))  # (B, H)
        x = F.relu(self.lin2(x))  # (B, H)
        x = torch.sum(x, dim=1)  # (B, 22)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic4(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, mode=1, edges=None):
        super().__init__()
        self.act_dim = act_dim
        self.mode = mode
        self.conv1 = GCNConv(in_channels, in_channels)
        # self.lin1 = nn.Linear(in_channels + self.mode + 1, hidden_size)
        if (mode == 0) | (mode == 1):
            self.lin1 = nn.Linear(in_channels + 1, hidden_size)
        else:
            self.lin1 = nn.Linear(in_channels + 2, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        concat = torch.cat([x, action], dim=-1)  # (B,N,22)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic5(nn.Module):
    """
    Architecture 5, GNN, Pointwise Multiplication, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6, edges=None):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        action = action + 1
        action = action.unsqueeze(-1)  # (B,N,1)
        x = x * action  # pointwise multiplication (B,N,21)
        x = F.relu(self.lin1(x))  # (B,N,H)
        x = F.relu(self.lin2(x))  # (B,N,H)
        x = x.sum(dim=1)  # (B,H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x
    

class GNNCritic6(nn.Module):
    """
    Architecture 6: OD-based action
    """

    def __init__(self, in_channels, hidden_size=128, act_dim=10, mode=1, edges=None):
        super().__init__()
        self.nregion = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + self.nregion, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.nregion, self.in_channels)  # (B,N,21)
        concat = torch.cat([x, action], dim=-1)  # (B,N,22)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x
    

class GNNCritic4_1(nn.Module):
    """
    Architecture 4: OD-based action with GNN
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, mode=1, edges=None):
        super().__init__()
        self.nregion = act_dim
        self.edges = edges
        self.conv1 = GCNConv(in_channels, in_channels)

        self.lin1 = nn.Linear(2*in_channels+1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.nregion, self.in_channels)  # (B,N,21)

        # Obtain edge features using 'out' for the updated node features
        # edge_features = torch.cat([x[edges[:, 0]], x[edges[:, 1]]], dim=1)
        edges_src = self.edges[:, 0] # [E]
        edges_dst = self.edges[:, 1] # [E]

        # Obtain features for each node involved in an edge
        edge_features_src = x[:, edges_src, : ] # [#batch, E, #features]
        edge_features_dst = x[:, edges_dst, : ] # [#batch, E, #features]

        # Concatenate features from source and destination nodes
        edge_features = torch.cat([edge_features_src, edge_features_dst], dim=2)

        concat = torch.cat([edge_features, torch.flatten(action,start_dim=-2).unsqueeze(-1)], dim=-1)  # (B,N,22)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x
    
class MLPCritic4(nn.Module):
    """
    Architecture 4: OD-based action with MLP
    """

    def __init__(self, in_channels, hidden_size=128, act_dim=10, mode=1, edges=None):
        super().__init__()
        self.nregion = act_dim
        self.lin1 = nn.Linear(self.nregion*(in_channels + self.nregion), hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        state = state.reshape(-1, self.nregion, self.in_channels)
        concat = torch.cat([state, action], dim=-1)  # (B,N,22)
        x = concat.reshape(-1, self.nregion*(self.in_channels + self.nregion))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = F.relu(self.lin3(x))  # (B, N, H)
        x = self.lin4(x).squeeze(-1)  # (B)
        return x
    
class MLPCritic4_1(nn.Module):
    """
    Architecture 4: Origin-based action with MLP
    """

    def __init__(self, in_channels, hidden_size=128, act_dim=10, mode=1, edges=None):
        super().__init__()
        self.nregion = act_dim
        if (mode == 0) | (mode == 1):
            self.lin1 = nn.Linear(self.nregion*(in_channels + 1), hidden_size)
            self.nfeatures = in_channels + 1
        else:
            self.lin1 = nn.Linear(self.nregion*(in_channels + 2), hidden_size)
            self.nfeatures = in_channels + 2
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        state = state.reshape(-1, self.nregion, self.in_channels)
        concat = torch.cat([state, action], dim=-1)  # (B,N,22)
        x = concat.reshape(-1, self.nregion * self.nfeatures)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = F.relu(self.lin3(x))  # (B, N, H)
        x = self.lin4(x).squeeze(-1)  # (B)
        return x
    

#########################################
######### VALUE FUNCTION ################
#########################################
class VF(torch.nn.Module):

    def __init__(self, in_channels=4, hidden_size=32, out_channels=1, nnodes=4):
        super(VF, self).__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.lin1 = nn.Linear(in_channels + hidden_size, hidden_size)
        self.g_to_v = nn.Linear(hidden_size, out_channels)
        self.nnodes = nnodes

    def forward(self, state, edge_index):
        x_pp = self.conv1(state, edge_index)
        x_pp = torch.cat([state, x_pp], dim=1)
        x_pp = x_pp.reshape(-1, self.nnodes, self.in_channels + self.hidden_size)
        v = torch.sum(x_pp, dim=1)
        v = F.relu(self.lin1(v))
        v = self.g_to_v(v)
        return v.squeeze(-1)