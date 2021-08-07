import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action) -> None:
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            # NOTICE: we would like the output to be (0,1) in our simulator
            # nn.Sigmoid()
            nn.Tanh()
        )
        
        self.max_action = max_action
    
    def forward(self, state):
        return self.max_action * self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))


class DDPG:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.action_dim = action_dim
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    # def random_action(self):
    #     action = np.random.uniform(0.,1.,self.action_dim)*self.max_action
    #     return action
    
    def train(self, replay_buffer, batch_size=256):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # compute the target Q
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done* self.discount * target_Q).detach()

        # get current Q estimation
        current_Q = self.critic(state, action)

        # compute critic loss/TD-loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)