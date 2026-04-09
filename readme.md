# OpenEnv Hackathon Submission
## Overview  
This project implements an adaptive decision support system for project delivery, modeled as a reinforcement learning-inspired environment. It simulates real-world project execution scenarios and applies dynamic decision-making to improve delivery outcomes.

The system continuously evaluates project state and recommends corrective actions such as task prioritization, workload optimization, and resource reallocation.

---

## Tasks  

The environment supports the following project delivery scenarios:

- delay_recovery  
  Simulates high-delay conditions where the objective is to reduce schedule slippage through prioritization and execution control.

- resource_crunch  
  Simulates constrained resource conditions where delivery improvement depends on reallocating and balancing limited capacity.

- priority_conflict  
  Simulates situations where sufficient resources exist but task prioritization is suboptimal, requiring better focus and workload alignment.

---

## Files  

- my_env_v4.py  
  Defines the environment, including project state, scenario setup, transition logic, and reward evaluation.

- inference.py  
  Implements the decision engine that evaluates the current state and selects corrective actions. Executes all supported scenarios sequentially.

- openenv.yaml  
  Configuration file defining environment metadata and task setup.

- Dockerfile  
  Containerization configuration for reproducible execution.

- requirements.txt  
  List of required dependencies.

---

## How It Works  

The system operates as a closed-loop decision process:

1. Initialize scenario-specific project conditions  
2. Observe current state (delay, resource availability)  
3. Select corrective action  
4. Apply action and update project state  
5. Evaluate outcome using a reward function  
6. Repeat until completion or maximum steps reached  

---

## Run  

### Build the Docker image

### Run the solution


This executes all three scenarios sequentially:
- delay_recovery  
- resource_crunch  
- priority_conflict  

---

## Expected Output  

The program produces structured logs for each scenario, including:

- Step-by-step actions selected  
- Reward at each step  
- Completion status  
- Final average score  

Each scenario runs independently and reports its outcome.

---

## Evaluation  

- Maximum of 8 steps per scenario  
- Reward generated at each step  
- Final score calculated as average reward  
- Success determined based on threshold score  

---

## Key Characteristics  

- Adaptive decision-making based on current execution state  
- Scenario-aware behavior using a single inference engine  
- Modular separation of environment and decision logic  
- Designed for extensibility to real-world project data  

---

## Future Enhancements  

- Integration with real project management tools (e.g., Jira, MS Project)  
- Transition to trained reinforcement learning models  
- Use of historical data for improved decision-making  
- Schedule prediction and risk forecasting  

---

## Summary  

This solution demonstrates how project delivery can be improved through adaptive, state-driven decision systems. By simulating multiple execution scenarios and applying dynamic corrective actions, it establishes a foundation for more intelligent and responsive delivery management.