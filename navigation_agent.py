import numpy as np
import random
from navigation_model import QNetwork
import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.9            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 0.001              # learning rate 
UPDATE_EVERY = 4        # how often to update the network
PRIORITIES_EPS = .01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(state_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, B):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, B)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, B):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            B (float) : controls sample weighting
        """
        states      = experiences['state']
        actions     = experiences['action']
        rewards     = experiences['reward']
        next_states = experiences['next_state']
        dones       = experiences['done']
        priorities  = experiences['priority']
        idx         = experiences['idx']
        # Get the greedy action based on the local policy
        Q_actions_next = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        # Evaluate actions using the target Q function
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_actions_next)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # sample weights
        w = (1/len(self.memory)/priorities)**B
        # TD - error
        TD_error = Q_targets - Q_expected
        new_priorities = torch.abs(TD_error).cpu().data.numpy() + PRIORITIES_EPS
        self.memory.max_priority = max(self.memory.max_priority, new_priorities.max())
        self.memory.priorities_sum += new_priorities.sum() - self.memory.experience['priority'].retrieve(idx).sum()
        self.memory.experience['priority'].update(new_priorities, idx)
        
        # Compute loss
        loss = torch.sum((TD_error*w)**2)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class Data:
    def __init__(self, shape, dtype):
        self.data = np.empty(shape, dtype)
        self.row = 0
        
    def get(self):
        return self.data[:self.row]
        
    def retrieve(self, idx):
        data = self.get()
        return data[idx]
    
    def add(self, value):
        try:
            self.data[self.row] = value
            self.row +=1
        except IndexError:
            self.data[:-1] = self.data[1:]
            self.data[-1] = value
            
    def update(self, value, idx):
        self.data[idx] = value
        
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.N = []
        self.batch_size = batch_size
        self.experience = {"state" : Data( (buffer_size, state_size), np.float32),
                           "action": Data( (buffer_size, 1), np.int32), 
                           "reward": Data( (buffer_size, 1), np.int32), 
                           "next_state" : Data( (buffer_size, state_size), np.float32), 
                           "done" : Data( (buffer_size, 1), np.int32), 
                           "priority" : Data( (buffer_size, 1), np.float32)}
        
        self.max_priority = 1
        self.priorities_sum = 0
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.experience["state"].add(state)
        self.experience["action"].add(action)
        self.experience["reward"].add(reward)
        self.experience["next_state"].add(next_state)
        self.experience["done"].add(done)
        
        n = len(self)
        if n<self.buffer_size:
            self.N.append(n)
            self.priorities_sum += self.max_priority
        else:
            self.priorities_sum += self.max_priority - self.experience["priority"].get()[0]
        self.experience["priority"].add(self.max_priority)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory according to piorities."""
        p = self.experience["priority"].get()[:,0]/self.priorities_sum
        idx = np.random.choice(self.N, size=self.batch_size, p=p, replace=False)
        output = {}
        data = self.experience['state'].retrieve(idx)
        output['state']      = torch.from_numpy(data).float().to(device)
        data = self.experience['action'].retrieve(idx)
        output['action']     = torch.from_numpy(data).long().to(device)
        data = self.experience['reward'].retrieve(idx)
        output['reward']     = torch.from_numpy(data).float().to(device)
        data = self.experience['next_state'].retrieve(idx)
        output['next_state'] = torch.from_numpy(data).float().to(device)
        data = self.experience['done'].retrieve(idx)
        output['done'] = torch.from_numpy(data).float().to(device)
        data = self.experience['priority'].retrieve(idx)/self.priorities_sum
        output['priority'] = torch.from_numpy(data).float().to(device)
        output['idx']        = idx
        return output

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.N)
    
    
