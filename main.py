import numpy as np
import gym
import petname
from sys import maxint
from scipy.stats import norm
from types import IntType
import os


def run():
  
    env_name = 'MountainCar-v0'
    run_name = petname.Name()
    print "Starting environment: {0}, run: {1}".format(env_name, run_name)
    agent_type = 'ContinuousSarsaLambda'
    env = gym.make(env_name)
    env.monitor.start('runs/{0}/{1}/{2}'.format(env_name, agent_type, run_name))
    if agent_type == 'SarsaLambda':
        sarsa_agent = SarsaLambdaAgent(env)
    elif agent_type == 'Sarsa':
        sarsa_agent = SarsaAgent(env)
    elif agent_type == 'QLearner':
        sarsa_agent = QLearner(env)
    elif agent_type == 'ContinuousSarsaLambda':
        sarsa_agent = ContinuousSarsaLambdaAgent(env)
    else:
        raise NotImplementedError

    fname = 'runs/{0}/{1}/{2}/q_values'.format(env_name, agent_type, run_name)
    if not os.path.exists(fname):
        os.makedirs(fname)
    save_every = 10
    np.save('{0}/init'.format(fname), sarsa_agent.q)
    

    run_episodes(sarsa_agent, env, episodes=3000, max_steps=env.spec.timestep_limit,
                                                                save_every=save_every, fname=fname)


def run_episodes(agent, env, episodes=100, max_steps=1000, save_every=None, fname=None):

    if fname is not None:
        assert type(save_every) is IntType

    if max_steps is None or max_steps < 0:
        max_steps = maxint

    for i_episode in range(episodes):
        observation = env.reset()
        agent.set_new_episode()
        action = agent.action_from_observation(observation)

        for i in xrange(max_steps):
            env.render()
            observation_prime, reward, done, info = env.step(action)
            action = agent.learn(observation, observation_prime, action, reward, done)
            observation = observation_prime
            if done:
                print "Episode Done"
                break
        if fname is not None and (i_episode + 1) % save_every == 0:
            np.save('{0}/epoch_{1}'.format(fname, i_episode+1), agent.q)


    env.monitor.close()


class SarsaAgent(object):

    def __init__(self, environment, policy = "e-greedy", alpha0=.5, gamma=1,
                                            alpha_schedule={'type':'constant', 'parameters':None}):
        self.environment = environment

        if policy == 'e-greedy':
            self.take_action = self.e_greedy_action
        elif policy == 'greedy':
            self.take_action = self.greedy_action

        self.iteration = 0
        self.alpha0 = alpha0
        self.alpha = alpha0
        self.parse_alpha_schedule(alpha_schedule)
        self.gamma = gamma
        self.actions = np.array([0,1,2])
        self.set_up_tiles()
        self.initialize_q()
        self.take_action_in_learn = self.take_action

    def parse_alpha_schedule(self, schedule):

        if schedule['type'] == 'exp':
            self.alpha_step = self.exp_decay(schedule['parameters'][0])
        elif schedule['type'] == 'multiplication':
            self.alpha_step = self.multiplicaton_decay(schedule['parameters'][0])
        elif schedule['type'] == 'constant':
            self.alpha_step = lambda: self.alpha0
        elif schedule['type'] == 'itdecay':
            self.alpha_step = self.get_itdecay(schedule['parameters'][0])
        else:
            raise NotImplementedError
    def get_itdecay(self, c):
        return lambda: self.alpha0*c *1./self.iteration        

    def multiplication_decay(self, r):
        def cooling():
            return self.alpha0*r**self.iteration
        return cooling

    def exp_decay(self, r):
        def cooling():
            return self.alpha0*10**(-self.iteration*1./r)
        return cooling

    def set_up_tiles_normal_distrub(self, n_tiles=30, sigma=0.5, mu=-0.45, n_speed=10):
        distrub = norm(loc=mu, scale=sigma)
        low = distrub.cdf(-1.2)
        high = distrub.cdf(0.6)
        uniform_buckets = np.linspace(low, high, n_tiles)
        self.tiles = distrub.ppf(uniform_buckets)

        self.speed = np.linspace(-0.07, 0.07, n_speed)

    def initialize_q_normal_distrub(self, q0=0):
        self.q = np.empty((self.tiles.size, self.speed.size, self.actions.size))
        self.q.fill(q0)

    def get_state_normal_distrub(self, observation):

        return np.array([np.digitize(observation[0], self.tiles),
                np.digitize(observation[1], self.speed)])


    def set_up_tiles(self, n_tiles=[9,9]):
        self.tiles = n_tiles[0]
        self.speed = n_tiles[1]
        self.tile_size = (self.environment.observation_space.high -\
                                self.environment.observation_space.low)*1./ np.array(n_tiles)

    def initialize_q(self, q0=0):
        self.q = np.empty((self.tiles, self.speed, self.actions.size))
        self.q.fill(q0)


    def get_state(self, observation):
        active_tiles = ((observation-self.environment.observation_space.low)
                                                                    /self.tile_size).astype(int)
        return active_tiles

    def set_new_episode(self):
        self.iteration = 1


    def action_from_observation(self, observation):
        state = self.get_state(observation)
        return self.take_action(state)

    def greedy_action(self, state):
        if np.unique(self.q[state[0], state[1]]).size ==  1:
            return np.random.choice(self.actions)
        return np.argmax(self.q[state[0], state[1]])

    def e_greedy_action(self, state, e=0.2):
        draw = np.random.rand()
        if draw > e:
            return np.argmax(self.q[state[0], state[1]])
        return np.random.choice(self.actions)

    def update(self, s, s_prime, a, a_prime, reward, done_prime):
        self.q[s[0], s[1], a] += self.alpha*(reward +
                                (1-done_prime)*self.gamma*self.q[s_prime[0], s_prime[1], a_prime] -
                                                                            self.q[s[0], s[1],a])
    def learn(self, observation, observation_prime, action, reward, done_prime):

        s = self.get_state(observation)
        s_prime = self.get_state(observation_prime)
        a_prime = self.take_action_in_learn(s_prime)
        self.update(s, s_prime, action, a_prime, reward, done_prime)
        self.alpha = self.alpha_step()
        self.iteration += 1
        return self.take_action(s_prime)


class SarsaLambdaAgent(SarsaAgent):

    def __init__(self, environment, policy="greedy", lamb=0.9, gamma=1, alpha0=.5,
                                            alpha_schedule={'type':'constant', 'parameters':None},
                                            eligibility_trace='replace'):
        super(SarsaLambdaAgent,self).__init__(environment, policy=policy, gamma=gamma,
                                                                    alpha0=alpha0,
                                                                    alpha_schedule=alpha_schedule) 
        self.lamb = lamb
        self.e = np.empty(self.q.shape)
        self.eligibility_trace = eligibility_trace

    def set_new_episode(self):
        super(SarsaLambdaAgent,self).set_new_episode()
        self.e.fill(0)

    def update(self, s, s_prime, a, a_prime, reward, done_prime):
        delta = reward + (1-done_prime)*self.gamma*self.q[s_prime[0], s_prime[1], a_prime] -\
                                                                                self.q[s[0], s[1],a]

        if self.eligibility_trace == 'replace':
            self.e[s[0],s[1],a] = 1
        elif self.eligibility_trace == 'cumulative':
            self.e[s[0], s[1], a] += 1
        else:
            raise NotImplementedError
        self.q += self.alpha*delta*self.e
        self.e *= self.gamma*self.lamb

        return a_prime


class ContinuousSarsaLambdaAgent(SarsaLambdaAgent):
    def __init__(self, environment, policy = "greedy", alpha0=.005, gamma=1, lamb=0.9,
                                            alpha_schedule={'type':'constant', 'parameters':[0.1]}):

        super(ContinuousSarsaLambdaAgent,self).__init__(environment, policy=policy, gamma=gamma,
                                                                    alpha0=alpha0,
                                                                    lamb=lamb,
                                                                    alpha_schedule=alpha_schedule)

    def set_up_tiles(self):
        self.n_tiles = np.array([9,9])
        self.offset_fraction = 0.5
        self.n_features = 10
        self.tile_size = (self.environment.observation_space.high -\
                                self.environment.observation_space.low)*1./ np.array(self.n_tiles)

        self.offsets = np.random.uniform(low=-self.offset_fraction*self.tile_size,
                                        high=self.offset_fraction*self.tile_size,
                                        size=(self.n_features, self.tile_size.size))
          
    def get_state(self, observation):
        f = np.zeros(self.n_features*np.prod(self.n_tiles))

        active_tiles = ((observation-self.offsets-self.environment.observation_space.low) /self.tile_size).astype(int)


        idx = (active_tiles[:,0]*self.n_tiles[1]+active_tiles[:,1]) +\
                                                    np.arange(self.n_features)*np.prod(self.n_tiles)


        assert np.sum(idx<0) == 0
        f[idx] = 1
        return f

    def greedy_action(self, state):
        scores = np.dot(self.q, state)
        return np.random.choice(np.argwhere(scores==np.amax(scores)).flatten())


    def e_greedy_action(self, state, e=0.1):
        draw = np.random.rand()
        if draw > e:
            return self.greedy(state)
        return np.random.choice(self.actions)

    def initialize_q(self, q0=0):
        self.q = np.empty((3,self.n_features*np.prod(self.n_tiles)))
        self.q.fill(q0)
        self.b = np.zeros(3)

    def update(self, s, s_prime, a, a_prime, reward, done_prime):

        delta = reward + (1-done_prime)*self.gamma*np.dot(self.q[a_prime], s_prime)  - np.dot(self.q[a], s)

        if self.eligibility_trace == 'replace':
            self.e[a,s==1] = 1
        elif self.eligibility_trace == 'cumulative':
            self.e[a,s==1] += 1
        else:
            raise NotImplementedError


        self.q += self.alpha*delta*self.e
        self.e *= self.gamma*self.lamb

class ContinuousQLearner(ContinuousSarsaLambdaAgent):
    def __init__(self, environment, policy = "e-greedy", alpha0=.1, gamma=1,
                                            alpha_schedule={'type':'constant', 'parameters':None}):

        super(ContinuousQLearner,self).__init__(environment, policy=policy, gamma=gamma,
                                                                    alpha0=alpha0,
                                                                    alpha_schedule=alpha_schedule)
        self.take_action_in_learn = self.greedy_action



class QLearner(SarsaLambdaAgent):

    def __init__(self, environment, policy="greedy", lamb=0.9, gamma=0.5, alpha0=1,
                                            alpha_schedule={'type':'constant', 'parameters':None},
                                            eligibility_trace='replace'):
        
        super(QLearner,self).__init__(environment, policy=policy, gamma=gamma, alpha0=alpha0,
                                                             alpha_schedule=alpha_schedule,
                                                             eligibility_trace=eligibility_trace)
        self.take_action_in_learn = self.take_greedy

    def take_greedy(self, state):
        return np.argmax(self.q[state[0], state[1]])




if __name__ == '__main__':
    run()