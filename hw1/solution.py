#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from network import *
from utils import *


class Config:
    def __init__(self):
        self.hidden_sizes = [50, 25]
        self.learning_rate = 2e-3

        self.batch_size = 32
        self.epochs = 25

        self.cv = 1
        self.iters = 3
        
        # DAgger
        self.d_rollouts = 20

        
def initialize(env_name):
    print('----------Environments:%s----------'%env_name)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    args.envname = env_name
    args.expert_policy_file = 'experts/' + env_name +'.pkl'
    
    # Load Expert Policy
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    
    # Load Expert Datas
    print('loading expert data ...')
    with open(os.path.join('data/expert_data', args.envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)
    print('load finished')
    
    # Initialize Envrionments
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    return env, args, expert_data, policy_fn

    
def preprocess(env, expert_data):
    input_size  = env.observation_space.shape[0]
    output_size = env.action_space.low.shape[0]
    N = expert_data['actions'].shape[0]
    expert_data['actions'] = expert_data['actions'].reshape(-1, output_size)
    
    X = expert_data['observations']
    y = expert_data['actions']
    
    return X, y


def train(agent, X, y, config, sess, env_name):
    print('--Network Training Start--')
    writer = tf.summary.FileWriter('./logs/%s/'%(env_name), sess.graph)

    total_batch = X.shape[0] // config.batch_size
    for e in range(config.epochs):
        print('Epoch:',(e+1))
        X, y = shuffle_data(X, y)

        for batch_idx in range(total_batch):
            batch_x = X[batch_idx*config.batch_size:(batch_idx+1)*config.batch_size]
            batch_y = y[batch_idx*config.batch_size:(batch_idx+1)*config.batch_size]

            summary = agent.train(sess, batch_x, batch_y)
            writer.add_summary(summary, global_step = sess.run(agent.global_step))
    writer.close()
    print('Network Training Finished')



def roll_out(args, agent, max_timesteps, sess, config, it):
    print('---Start Rolling out Agent---')
    
    returns = []
    observations = []
    actions = []

    env = gym.make(args.envname)
    for i in range(config.d_rollouts):
        #print('iter', i)
        obs  = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = agent.predict(sess, obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            #if steps % 100 == 0: print("%i/%i"%(steps, max_timesteps))
            if steps >= max_timesteps:
                break
        returns.append(totalr)
    #print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    agent_data = {'observations': np.array(observations),
                   'actions': np.array(actions),
                   'returns':np.array(returns)}

    with open(os.path.join('data/agent_data', args.envname + '_h' + str(config.hidden_sizes[0]) +
                           '_iter' + str(it) + '.pkl'), 'wb') as f:
        pickle.dump(agent_data, f, pickle.HIGHEST_PROTOCOL)       

    print('Roling Finished')
    
    return agent_data


def aggregate(X, y, agent_data, policy_fn):
    print('Start Aggretation')
    actions = []
    for obs in agent_data['observations']:
        action = policy_fn(obs[None, :])
        actions.append(action)
        
    X = np.vstack([X, agent_data['observations']])
    y = np.vstack([y, np.array(actions).reshape(-1, y.shape[1])])
    print('Aggregation Finished')
    
    return X, y
    

def main():
    config = Config()
    
    #envnames = ['Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','Reacher-v2','Walker2d-v2']
    envnames = ['Walker2d-v2']
    
    for envname in envnames:
        tf.reset_default_graph()
    
        env, args, expert_data, expert_policy = initialize(envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        X, y = preprocess(env, expert_data)

        CloneNet = CloneNetwork(X.shape[1], y.shape[1], config)
        
        # Train Behavioral Cloning Network
        with tf.Session() as sess:
            tf_util.initialize()

            # config.iter == 0: Behavioral Cloning
            # config.iter  > 0: DAgger
            for it in range(config.iters):
                train(CloneNet, X, y, config, sess, envname)

                #Start Roll-Out
                agent_data = roll_out(args, CloneNet, max_steps, sess, config, it)

                # Aggregate
                X, y = aggregate(X, y, agent_data, expert_policy)

if __name__ == '__main__':
    main()
