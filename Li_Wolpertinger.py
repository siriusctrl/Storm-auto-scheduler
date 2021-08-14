import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            # NOTICE: we would like the output to be (0,1) in our simulator
            # nn.Sigmoid()
            nn.Tanh()
        )
        
    
    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim+action_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))


class Wolpertinger:
    def __init__(self, state_dim, action_dim, n_machines, n_exe, discount=0.99, tau=0.01):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.action_dim = action_dim

        self.n_machines = n_machines
        self.n_exe = n_exe

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        proto_action = self.actor(state).cpu().data.numpy().flatten()
        # NOTICE: we add fixed exploreation noise to the proto-action which is different than original paper that use a decrasing one
        if noise:
            proto_action += 0.1*np.random.normal(0, 1, size=proto_action.shape)
        action = self.proto_to_action(proto_action)
        return action, proto_action
    
    def proto_to_action(self, proto_action):
        proto_action = proto_action.reshape((self.n_exe, self.n_machines))
        action = np.zeros(proto_action.shape)
        col = np.argmax(proto_action, axis=1)
        row = np.array(range(self.n_exe))
        action[row, col] = 1
        return action.flatten()
    
    def train(self, replay_buffer, batch_size=32):
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

if __name__ == '__main__':
    model = Wolpertinger(20*5+2, 20*5, 5, 20)
    test = np.random.randn(20,5)
    res = np.zeros(test.shape)
    test = np.argmax(test, axis=1)
    row = np.array(range(len(test)))
    res[row, test] = 1
    # print(test)
    # print(res)
    res = np.concatenate((res.flatten(), [2,2]))
    action, proto_action = model.select_action(res)
    print(sum(action))
    print(action)