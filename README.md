# Multi-Action Coordination Game: Limited Sum

An implementation and analysis of a **Limited Sum Coordination Game** using Game Theory and Reinforcement Learning.

## The Game
The game is a symmetric static game where two players choose an integer from $S = \{0, 1, 2, 3, 4, 5\}$.
- If $sum(choices) \le 5$: Each player gets their requested amount.
- If $sum(choices) > 5$: Both players receive 0.

This creates a tension between efficiency (sum = 5) and equity, leading to complex social dynamics in iterated versions.

## Features
- **Core Engine**: Object-oriented implementation of game rules, matches, and round-robin tournaments.
- **Agent Ecosystem**:
    - **Traditional**: AlwaysK, UniformRandom, Tit-For-Tat.
    - **SherlockHolmes**: A rule-based heuristic agent designed to detect patterns and punish greed.
    - **DQNAgent**: A Reinforcement Learning agent using a **Deep Q-Network** with Replay Buffer and Target Networks.
- **Evolutionary Simulations**: Analysis of how populations of strategies evolve over generations based on fitness (payoffs).

## Reinforcement Learning Approach
The project features a **DQN (Deep Q-Network)** implementation using PyTorch to solve the coordination dilemma. 
- **State Space**: Encodes the last $k$ rounds and opponent context (mean/variance).
- **Architecture**: Multi-Layer Perceptron (MLP) trained with experience replay.

## Installation
```bash
git clone [https://github.com/youruser/multi-action-coordination-game.git](https://github.com/youruser/multi-action-coordination-game.git)
cd multi-action-coordination-game
pip install -r requirements.txt
