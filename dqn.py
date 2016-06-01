import theano
from theano import tensor as T
import petname
import gym
import os
import numpy as np
from types import IntType

def run():
  
    env_name = 'MountainCar-v0'
    run_name = petname.Name()
    print "Starting environment: {0}, run: {1}".format(env_name, run_name)
    agent_type = 'DQNAgent'
    env = gym.make(env_name)
    env.monitor.start('runs/{0}/{1}/{2}'.format(env_name, agent_type, run_name))
    if agent_type == 'DQNAgent':
        sarsa_agent = DQNAgent(env)
    else:
        raise NotImplementedError

    fname = 'runs/{0}/{1}/{2}/q_values'.format(env_name, agent_type, run_name)
    if not os.path.exists(fname):
        os.makedirs(fname)
    save_every = 10
    run_episodes(sarsa_agent, env, episodes=3000, max_steps=env.spec.timestep_limit,
                                                                save_every=save_every, fname=fname)

def run_episodes(agent, env, episodes=100, max_steps=1000, save_every=None, fname=None):

    if fname is not None:
        assert type(save_every) is IntType

    if max_steps is None or max_steps < 0:
        max_steps = maxint

    for i_episode in range(episodes):
        observation = env.reset()
        action = agent.take_action(observation)

        for i in xrange(max_steps):
            env.render()
            observation_prime, reward, done, info = env.step(action)
            agent.learn(observation, observation_prime, action, reward, done)
            observation = observation_prime
            action = agent.take_action(observation)
            if done:
                print "Episode Done"
                break
        print agent.replay_memory.shape
        #if fname is not None and (i_episode + 1) % save_every == 0:
        #    np.save('{0}/w_epoch_{1}'.format(fname, i_episode+1), agent.approximator.w)
        #    np.save('{0}/b_epoch_{1}'.format(fname, i_episode+1), agent.approximator.b)



    env.monitor.close()

class DQNAgent(object):

    def __init__(self, environment, policy="e-greedy", lamb=0.9, gamma=1, alpha0=.1,
                                                                        weight_constant_step=1000,
                                                                        replay_memory_size=1e5,
                                                                        minibatch_size=10):

        if policy == 'e-greedy':
            self.take_action = self.e_greedy_action
        elif policy == 'greedy':
            self.take_action = self.greedy_action
        else:
            raise NotImplementedError

        self.replay_memory_size = replay_memory_size
        self.replay_memory = np.empty((0,7))
        self.weight_constant_step = weight_constant_step

        self.iteration = 0
        self.alpha = alpha0
        self.gamma = gamma
        self.actions = np.array([0,1,2])
        self.activation_names = ['tanh', 'tanh', 'linear']
        self.n_hidden = [20, 10]

        self.approximator = PerceptronApproximator(n_hidden=self.n_hidden,
                                                            activation_names=self.activation_names)
        self.frozen_approximator = PerceptronApproximator(n_hidden=self.n_hidden,
                                                            activation_names=self.activation_names)
        for p in self.frozen_approximator.params:
            p.set_value(p.get_value()*0)
        self.list_index = 0
        self.minibatch_size = minibatch_size
        self.iteration = 1

    def greedy_action(self, state, approximator=None):
        if len(state.shape) == 1:
            state = state.reshape(1,-1)

        if approximator is None:
            approximator = self.approximator
        q = approximator.evaluate(state)
        return np.random.choice(np.argwhere(q==np.amax(q)).flatten())

    def e_greedy_action(self, state, e=0.2, approximator=None):
        if len(state.shape) == 1:
            state = state.reshape(1,-1)

        if approximator is None:
            approximator = self.approximator
        draw = np.random.rand()
        if draw > e:
            return self.greedy_action(state)
        return np.random.choice(self.actions)

    def sample(self):
        idx = np.random.permutation(self.replay_memory.shape[0])
        return self.replay_memory[idx[:min(self.replay_memory.shape[0], self.minibatch_size)]]

    def learn(self, observation, observation_prime, action, reward, done_prime):
        new_sample = np.array([list(observation) +[action, reward, done_prime] + list(observation_prime)])
        self.push_sample(new_sample)

        minibatch = self.sample()
        
        s = minibatch[:,:2]
        s_prime = minibatch[:,5:]
        r = minibatch[:,3].reshape(-1,1)
        d = minibatch[:,4].reshape(-1,1)
        a = np.asarray(minibatch[:,2], dtype=int)

        delta = r + (1-d)*self.gamma*np.max(self.frozen_approximator.evaluate(s_prime), axis=0)
        tmp = self.approximator.evaluate(s)
        self.approximator.train(s, delta, a, self.alpha)

        if self.iteration % self.weight_constant_step == 0:
            self.frozen_approximator.freeze(self.approximator)
            print "Weights frozen"
        self.iteration += 1

    def push_sample(self, new_sample):
        if np.any(np.sum(np.isclose(new_sample, self.replay_memory, atol=0.10), axis=1) == new_sample.shape[1]):
            return

        if self.replay_memory.shape[0] < self.replay_memory_size:

            self.replay_memory = np.vstack((self.replay_memory, new_sample))
        else:

            self.replay_memory[self.list_index,:] = new_sample
            self.list_index += 1
            self.list_index %= self.replay_memory_size

class HiddenLayer(object):

    def __init__(self, input_var, n_in, n_out, activation, n_actions=3, input_layer=False):
        
        self.input = input_var
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        self.w = theano.shared(value=get_weight_init_values(n_actions, self.n_in, self.n_out,
                                                                        activation=self.activation))
        self.b = theano.shared(value=np.zeros((n_actions,), dtype=theano.config.floatX))
        self.params = [self.w, self.b]

        def step(w, b, x):
            a = T.dot(x,w.T) + b
            return self.activation(a)

        non_seq = []
        seq = [self.w, self.b]
        if input_layer:
            non_seq = [input_var]
        else:
            seq += [input_var]

        self.output, _ = theano.scan(step, sequences=seq, non_sequences=non_seq)


class OutputLayer(object):

    def __init__(self, input_var, n_in, n_out, activation, n_actions=3):
        self.input = input_var
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.w = theano.shared(value=get_weight_init_values(n_actions, self.n_in, self.n_out,
                                                                        activation=self.activation))
        self.b = theano.shared(value=np.zeros((n_actions,), dtype=theano.config.floatX))

        self.params = [self.w, self.b]
        def step(w, b, x):
            a = T.dot(x,w.T) + b
            return self.activation(a)

        seq = [self.w, self.b, input_var]

        self.output, _ = theano.scan(step, sequences=seq)


class PerceptronApproximator(object):

    def __init__(self, n_in=2, n_hidden=[10], n_out=1, activation_names=['sigmoid', 'linear']):

        self.x = T.matrix('x')
        self.y = T.matrix('y')
        self.action = T.lvector('action')
        self.lr = T.scalar('lr')
        self.n_hidden = np.array(n_hidden)
        self.n_in = n_in
        self.n_out = n_out
        self.activations = parse_activations(activation_names)
        self.build_graph()

    def freeze(self, approx):
        for w_old, w in zip(self.params, approx.params):
            w_old.set_value(w.get_value(borrow=False))

    def update_weights(self):
        self.w_old.set_value(self.w.get_value(borrow=False))
        self.b_old.set_value(self.b.get_value(borrow=False))

    def build_graph(self):

        self.hidden_layers = [None]*self.n_hidden.size
        self.params = []

        for i, h in enumerate(self.n_hidden):
            if i==0:
                self.hidden_layers[i] = HiddenLayer(self.x, self.n_in, h, self.activations[i],
                                                                    n_actions=3, input_layer=True)
            else:
                self.hidden_layers[i] = HiddenLayer(self.hidden_layers[i-1].output,
                                                    self.n_hidden[i-1], h,
                                                    self.activations[i],
                                                    n_actions=3, input_layer=False)

            self.params += self.hidden_layers[i].params

        self.output_layer = OutputLayer(self.hidden_layers[-1].output,
                                        self.n_hidden[-1], self.n_out, self.activations[-1])

        self.params += self.output_layer.params
        self.output = self.output_layer.output

        delta = self.y - self.output[self.action]
        self.loss = T.mean(delta**2)

        self.gparams = [T.grad(self.loss, param) for param in self.params]
        upd = [(param, param - self.lr*gth) for param, gth in zip(self.params, self.gparams)]
        
        self.evaluate = theano.function(inputs=[self.x], outputs=self.output)

        self.train = theano.function(inputs=[self.x, self.y, self.action, self.lr],
                                                                    outputs=self.loss, updates=upd)

def parse_activations(activation_list):
    """From list of activation names for each layer return a list with the activation functions"""

    activation = [None]*len(activation_list)
    for i, act in enumerate(activation_list):
        activation[i] = get_activation_function(act)

    return activation

def get_activation_function(activation):

    activations = ['tanh', 'sigmoid', 'relu', 'linear']

    if activation == activations[0]:
        return T.tanh
    elif activation == activations[1]:
        return T.nnet.sigmoid
    elif activation == activations[2]:
        return lambda x: x * (x > 0)
    elif activation == activations[3]:
        return lambda x: x
    else:
        raise NotImplementedError, \
        "Activation function not implemented. Choose one out of: {0}".format(activations)

def get_weight_init_values(n_actions, n_in, n_out, activation=None, rng=None):
    
    if rng is None:
        rng = np.random.RandomState(0)

    W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_actions, n_out, n_in)
                ),
                dtype=theano.config.floatX
            )
    if activation == theano.tensor.nnet.sigmoid:
        W_values *= 4
    return W_values
if __name__ == '__main__':
    run()
