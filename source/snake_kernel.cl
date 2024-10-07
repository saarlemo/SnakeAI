#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// Define constants for neural network and game parameters
#define INPUT_SIZE 12
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 3
#define N_HIDDEN 3
#define GRID_WIDTH 20
#define GRID_HEIGHT 20
#define MAX_SNAKE_LENGTH (GRID_WIDTH * GRID_HEIGHT)
#define MAX_STEPS 500
#define BONUS_STEPS 100

// Direction vectors: 0 - left, 1 - up, 2 - right, 3 - down
const int dx[4] = { -1, 0, 1, 0 };
const int dy[4] = { 0, -1, 0, 1 };

// Linear Congruential Generator for pseudo-random numbers
inline unsigned int rand_lcg(unsigned int *seed) {
    *seed = (1103515245 * (*seed) + 12345) & 0x7FFFFFFF;
    return *seed;
}

// Function to check collision in a given direction
inline int check_collision(int dir, int head_x, int head_y, int* snake_x, int* snake_y, int snake_length) {
    int nx = head_x + dx[dir];
    int ny = head_y + dy[dir];

    // Check wall collision
    if (nx < 0 || nx >= GRID_WIDTH || ny < 0 || ny >= GRID_HEIGHT)
        return 1;

    // Check self-collision
    for (int i = 0; i < snake_length; i++) {
        if (snake_x[i] == nx && snake_y[i] == ny)
            return 1;
    }

    return 0;
}

// Activation function: ReLU
inline float activation(float x) {
    return max(0.f, x);
}

void __kernel fitness_kernel(__global float* weights, __global float* fitness_values, int numGenomes, int numWeights) {
    int gid = get_global_id(0);

    int idxOffset = gid * numWeights;

    // Calculate indices for weight segments
    int idx = idxOffset; // Beginning index of input-hidden layer weights

    int idx_w_input_hidden = idx;

    idx += INPUT_SIZE * HIDDEN_SIZE;
    int idx_w_hidden_hidden = idx; // Beginning index of first hidden-hidden layer weights

    idx += (N_HIDDEN - 1) * HIDDEN_SIZE * HIDDEN_SIZE;
    int idx_w_hidden_output = idx; // Beginning index of hidden-output layer weights

    // Initialize game state variables
    int snake_x[MAX_SNAKE_LENGTH];
    int snake_y[MAX_SNAKE_LENGTH];
    int snake_length = 1;
    int direction = 2; // Initial direction: right
    int food_x, food_y;
    int steps = 0;
    int alive = 1;
    float score = 0.0f;

    // Initialize snake position at the center
    snake_x[0] = GRID_WIDTH / 2;
    snake_y[0] = GRID_HEIGHT / 2;

    // Initialize random seed based on weights (simple method)
    unsigned int seed = (unsigned int)(weights[0] * 10000.0f + get_global_id(0));

    // Initialize food position
    food_x = rand_lcg(&seed) % GRID_WIDTH;
    food_y = rand_lcg(&seed) % GRID_HEIGHT;

    // Ensure food is not placed on the snake
    while (food_x == snake_x[0] && food_y == snake_y[0]) {
        food_x = rand_lcg(&seed) % GRID_WIDTH;
        food_y = rand_lcg(&seed) % GRID_HEIGHT;
    }

    // Neural network arrays
    float inputs[INPUT_SIZE];
    float hidden[N_HIDDEN][HIDDEN_SIZE];
    float outputs[OUTPUT_SIZE];

    // Snake positions are stored in local arrays
    int* snake_x_global = &snake_x[0];
    int* snake_y_global = &snake_y[0];

    int currentMaxSteps = MAX_STEPS;

    while (alive && steps < currentMaxSteps) {
        // Calculate possible directions
        int left_direction = (direction + 1) % 4;
        int right_direction = (direction + 3) % 4;

        // Set neural network inputs
        inputs[0] = (float)check_collision(left_direction, snake_x[0], snake_y[0], snake_x_global, snake_y_global, snake_length);
        inputs[1] = (float)check_collision(right_direction, snake_x[0], snake_y[0], snake_x_global, snake_y_global, snake_length);
        inputs[2] = (float)check_collision(direction, snake_x[0], snake_y[0], snake_x_global, snake_y_global, snake_length);
        inputs[3] = (direction == 0) ? 1.0f : 0.0f; // Moving left
        inputs[4] = (direction == 2) ? 1.0f : 0.0f; // Moving right
        inputs[5] = (direction == 1) ? 1.0f : 0.0f; // Moving up
        inputs[6] = (direction == 3) ? 1.0f : 0.0f; // Moving down
        inputs[7] = (food_x < snake_x[0]) ? 1.0f : 0.0f; // Food left
        inputs[8] = (food_x > snake_x[0]) ? 1.0f : 0.0f; // Food right
        inputs[9] = (food_y < snake_y[0]) ? 1.0f : 0.0f; // Food above
        inputs[10] = (food_y > snake_y[0]) ? 1.0f : 0.0f; // Food below
        inputs[11] = 1.0f;

        // Forward pass through neural network
        // First hidden layer
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float sum = 0.0f;
            for (int j = 0; j < INPUT_SIZE; j++) {
                sum += inputs[j] * weights[idx_w_input_hidden + i * INPUT_SIZE + j];
            }
            hidden[0][i] = activation(sum);
        }

        // Hidden layers
        for (int l = 1; l < N_HIDDEN; l++) {
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                float sum = 0.0f;
                for (int j = 0; j < HIDDEN_SIZE-1; j++) {
                    sum += hidden[l - 1][j] * weights[idx_w_hidden_hidden + (l - 1) * HIDDEN_SIZE * HIDDEN_SIZE + i * HIDDEN_SIZE + j];
                }
                sum += weights[idx_w_hidden_hidden + (l - 1) * HIDDEN_SIZE * HIDDEN_SIZE + (i+1) * HIDDEN_SIZE];
                hidden[l][i] = activation(sum);
            }
        }

        // Output layer
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float sum = 0.0f;
            for (int j = 0; j < HIDDEN_SIZE-1; j++) {
                sum += hidden[N_HIDDEN - 1][j] * weights[idx_w_hidden_output + i * HIDDEN_SIZE + j];
            }
            sum += weights[idx_w_hidden_output + (i+1) * HIDDEN_SIZE];
            outputs[i] = activation(sum);
        }

        // Determine action based on highest output
        int action = 0;
        float max_output = outputs[0];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (outputs[i] > max_output) {
                max_output = outputs[i];
                action = i;
            }
        }

        // Update direction based on action
        if (action == 0) {
            direction = (direction + 1) % 4; // Turn left
        }
        else if (action == 1) {
            direction = (direction + 3) % 4; // Turn right
        }
        // Else continue straight

        // Move the snake: shift positions
        for (int i = snake_length - 1; i > 0; i--) {
            snake_x[i] = snake_x[i - 1];
            snake_y[i] = snake_y[i - 1];
        }
        snake_x[0] += dx[direction];
        snake_y[0] += dy[direction];

        // Check for collisions
        if (snake_x[0] < 0 || snake_x[0] >= GRID_WIDTH || snake_y[0] < 0 || snake_y[0] >= GRID_HEIGHT) {
            alive = 0;
            break;
        }
        for (int i = 1; i < snake_length; i++) {
            if (snake_x[0] == snake_x[i] && snake_y[0] == snake_y[i]) {
                alive = 0;
                break;
            }
        }
        if (!alive)
            break;

        // Check for food consumption
        if (snake_x[0] == food_x && snake_y[0] == food_y) {
            if (snake_length < MAX_SNAKE_LENGTH)
                snake_length++;
            score += 1.0f;
            currentMaxSteps += BONUS_STEPS;

            // Place new food
            int placed = 0;
            while (!placed) {
                food_x = rand_lcg(&seed) % GRID_WIDTH;
                food_y = rand_lcg(&seed) % GRID_HEIGHT;
                placed = 1;
                for (int i = 0; i < snake_length; i++) {
                    if (snake_x[i] == food_x && snake_y[i] == food_y) {
                        placed = 0;
                        break;
                    }
                }
            }
        }

        steps++;
    }

    double a_d = (double)score;
    double s_d = (double)steps;

    double fitnessVal = log1p(pow(2.0, a_d) + 500.0 * pow(a_d, 2.1) - pow(a_d, 1.2) * pow(0.25 * s_d, 1.3));

    fitness_values[gid] = (float)fitnessVal; // Result in float
}
