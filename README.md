# MONTE CARLO CONTROL ALGORITHM
## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm.

## PROBLEM STATEMENT
Develop a Python program that implements a Monte Carlo control algorithm to find the optimal policy for navigating the FrozenLake environment. The program should initialize the environment, define parameters (discount factor, learning rate, exploration rate), and implement decay schedules for efficient learning. It must generate trajectories based on an epsilon-greedy action selection strategy and update the action-value function using sampled episodes. Evaluate the learned policy by calculating the probability of reaching the goal state and the average undiscounted return. Finally, print the action-value function, state-value function, and optimal policy.

## MONTE CARLO CONTROL ALGORITHM
Initialize Q(s, a) arbitrarily for all state-action pairs

Initialize returns(s, a) to empty for all state-action pairs

Initialize policy π(s) to be arbitrary (e.g., ε-greedy)

For each episode:
a. Generate an episode using policy π
b. For each state-action pair (s, a) in the episode:
       i. Calculate G (return) for that (s, a) pair
       ii. Append G to returns(s, a)
       iii. Calculate the average of returns(s, a)
       iv. Update Q(s, a) using the average return
c. Update policy π(s) based on Q(s, a)

## MONTE CARLO CONTROL FUNCTION
```
def mc_control(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000,
               max_steps=200,
               first_visit=True):

    nS, nA = env.observation_space.n, env.action_space.n

    discounts = np.logspace(
        0, max_steps,
        num=max_steps, base=gamma,
        endpoint=False)

    alphas = decay_schedule(
        init_alpha, min_alpha,
        alpha_decay_ratio,
        n_episodes)

    epsilons = decay_schedule(
        init_epsilon, min_epsilon,
        epsilon_decay_ratio,
        n_episodes)

    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon:np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=bool)

        for t, (state, action, reward, _, _) in enumerate(trajectory):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    v = np.max(Q, axis=1)
    pi = np.argmax(Q, axis=1)
    return Q, v, pi, Q_track, pi_track
```
## OUTPUT:
**Name:** Keerthana Saravanan
**Register Number:** 212223240070
### Action value function
![image](https://github.com/user-attachments/assets/cce70db5-0004-4cf8-a027-bf074fe36011)

### Optimal value function
![image](https://github.com/user-attachments/assets/a956983d-c77a-4b4f-a829-d0a45fb9bf7c)

### Optimal policy
![image](https://github.com/user-attachments/assets/fa28dc76-0da7-4cca-b318-f6069888ae52)

### Success rate for the optimal policy
![image](https://github.com/user-attachments/assets/214b0511-2186-4c30-97ed-bc5723d2babd)


## RESULT:
Thus, to implement Monte Carlo Control to learn an optimal policy in a given environment and evaluate its performance in terms of goal-reaching probability and average return is executed successfully.
