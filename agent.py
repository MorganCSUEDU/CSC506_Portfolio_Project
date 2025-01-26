import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from config import GAMMA, LEARNING_RATE, BUFFER_SIZE, BATCH_SIZE, AGENT_CONFIG

def create_agent(env=None):
    from config import AGENT_CONFIG
    cfg = AGENT_CONFIG.copy()
    
    if env and cfg['state_dim'] is None:
        cfg['state_dim'] = env.num_nodes + 3
    
    return SimpleDQN(
        state_dim=cfg['state_dim'],
        num_actions=cfg['num_actions'],
        hidden_dim=cfg['hidden_dim']
    )

Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class SimpleDQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=64): 
        super().__init__()
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.num_actions)
        )
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
            
    def forward(self, x):
        return self.net(x)

class PrioritizedReplayBuffer:
    """Prioritized experience replay implementation"""
    def __init__(self, capacity=BUFFER_SIZE, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=BATCH_SIZE):
        if len(self.buffer) < batch_size:
            return None
            
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return indices, experiences, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

def train_step(agent, target_agent, optimizer, replay_buffer, gamma=GAMMA):
    sample = replay_buffer.sample()
    if sample is None:
        return 0.0, []

    indices, experiences, weights = sample
    states, actions, rewards, next_states, dones = zip(*experiences)
    states_t = torch.stack(states)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    next_states_t = torch.stack(next_states)
    dones_t = torch.tensor(dones, dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    rewards_t = rewards_t * 0.1  # Scale down rewards

    current_q = agent(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_actions = agent(next_states_t).argmax(dim=1)
        next_q = target_agent(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards_t + gamma * next_q * (1 - dones_t)

    td_errors = F.smooth_l1_loss(current_q, target_q, reduction='none')
    loss = (weights_t * td_errors).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)  # Tighter gradient clipping
    optimizer.step()

    return loss.item(), td_errors.detach().numpy()
