#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=200, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=8, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=2000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace):
    # Fix random seed
    np.random.seed(args.seed)

    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.

    discrete_gym = wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1"))   ## !!! number of states, seed?
    n_states  = discrete_gym.observation_space.n
    n_actions = discrete_gym.action_space.n

    Q = np.zeros((n_states, n_actions))   # pos_x, vel_x, pos_ang, vel_ang .... kazda 8 hodnot -> 8**4 stavu, akce 0, 1 - jed vlevo, jed vpravo  
    C = np.zeros((n_states, n_actions))


    for _ in range(args.episodes):
        # Perform episode, collecting states, actions and rewards
        states, actions, rewards = [], [], []
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Compute `action` using epsilon-greedy policy. Therefore,
            # with probability of `args.epsilon`, use a random action,
            # otherwise choose an action with maximum `Q[state, action]`.
            rnd = np.random.rand(1)[0]

            if rnd < args.epsilon:                      # random action
                action = np.random.randint(0,2)
            else:                                       # greedy action
                action = np.argmax(Q[state,:])              # actions are 0 or 1 for cartpole...

            # Perform the action.
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # TODO: Compute returns from the received rewards and update Q and C.
        G = 0     # return
        gamma = 1 # discount factor

        for state, action, reward in zip(states[::-1], actions[::-1], rewards[::-1]):
        # for state, action, reward in zip(states, actions, rewards):
            G = gamma*G + reward
            C[state, action] += + 1
            Q[state, action] += (1/C[state, action]) * (G - Q[state, action])



    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO: Choose a greedy action
            action = np.argmax(Q[state,:])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed)

    main(env, args)
