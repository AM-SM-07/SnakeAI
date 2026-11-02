@'
# 🐍 Snake AI using Deep Q-Learning (DQN)

An AI-powered Snake Game built using Python, Pygame and PyTorch.
The agent learns to play Snake through Deep Reinforcement Learning.

## 🚀 Project Overview

This project implements a Deep Q-Network (DQN) agent that learns
to navigate a Snake Game environment, collect food and avoid collisions.

## 🧠 Technologies Used

- Python 3.11
- PyTorch
- Pygame
- NumPy
- Matplotlib

## ⚙️ AI Architecture

- State representation: 11 features
- Neural network: 11 → 256 → 3
- Activation: ReLU
- Optimizer: Adam
- Loss function: Mean Squared Error
- Discount factor: 0.9
- Experience Replay: Implemented

## 📁 Project Structure

```text
SnakeAI/
├── agent.py
├── game.py
├── helper.py
├── model.py
├── requirements.txt
├── README.md
└── .gitignore