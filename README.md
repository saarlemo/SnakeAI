# SnakeAI
SnakeAI leverages a combination of genetic algorithms and fully connected neural networks to evolve intelligent agents capable of playing the classic Snake game.

## How to install
To train your own neural network, first copy the repository to a local folder. Open MATLAB (R2024a tested working) and run the file ```main.m```. The file contains explanations for all of the simulation parameters.

## How it works
### Classes
a. SnakeGame contains the logic of the game, state management and rendering.
b. Agent represents the AI agent that decides actions based on the game state.
c. Genome encodes the neural network's weights and architecture.
d. Population manages a collection of genomes (agents).
e. GeneticAlgorithm controls the evolutionary process of training AI agents.

### Neural network
The neural network inputs were selected to not depend on the size of the grid. This way a trained neural network can play games on a grid of arbitrary size. In total, there are 11 binary inputs:
1. ```1``` if turning left results in a collision; else ```0```.
2. ```1``` if turning right results in a collision; else ```0```.
3. ```1``` if continuing straight results in a collision; else ```0```.
4. ```1``` if the snake is currently moving left; else ```0```.
5. ```1``` if the snake is currently moving right; else ```0```.
6. ```1``` if the snake is currently moving up; else ```0```.
7. ```1``` if the snake is currently moving down; else ```0```.
8. ```1``` if food is to the left of the snake's head; else ```0```.
9. ```1``` if food is to the right of the snake's head; else ```0```.
10. ```1``` if food is to the above the snake's head; else ```0```.
11. ```1``` if food is to the below the snake's head; else ```0```.

The user can define the architecture freely, provided the input layer has 11 units and the output layer has 3 units. The outputs of each layer are processed through the ReLU activation function with an optional dropout mask being applied as in [2].

### Evolutionary process
1. **Initialization**: A population of agents (genomes) is created with each one having the pre-defined genome. If no pre-defined genome is supplied, each agent will be assigned a randomly generated one.
2. **Evaluation**: Each agent plays the Snake game, and its performance (fitness) is calculated based on the apples eaten and the survival time. By default, the fitness function of [1] is used. The function is designed to reward early exploring snakes, reward apples eaten and penalize steps taken. The user can also define a custom fitness function. The maximum amount of steps can (and should) be set to be finite, as otherwise some snakes might stay in an infinite loop circling around. To encourage eating apples rather than maximizing steps, each apple eaten can be set to give the agent more playtime.
3. **Selection**: $N$ agents with the highest fitness scores are selected as parents for the next generation.
4. **Crossover and mutation**: The selected parents produce offspring genomes through crossover (combining weights) and mutation (randomly altering weights).
5. **Iteration**: Steps 2-4 are repeated for a set number of generations, progressively evolving more competent AI agents.

### Visualization
The file ```main.m``` contains a section, where the most competent agent of the last generation can be visualized playing the Snake game.

## To do
-

## Citations
[1] [https://www.youtube.com/watch?v=vhiO4WsHA6c](https://www.youtube.com/watch?v=vhiO4WsHA6c)
[2] A. Sebastianelli, M. Tipaldi, S. L. Ullo and L. Glielmo, "A Deep Q-Learning based approach applied to the Snake game", *2021 29th Mediterranean Conference on Control and Automation (MED)*, PUGLIA, Italy, 2021, pp. 348-353, doi: 10.1109/MED51440.2021.9480232.