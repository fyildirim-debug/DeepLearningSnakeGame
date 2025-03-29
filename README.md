# AI Powered Snake Game

This project is an experiment bringing the classic snake game together with artificial intelligence. Using the Stable Baselines3 library and the DQN (Deep Q-Network) algorithm, the snake learns over time to eat food and avoid obstacles.

![Game Screenshot](placeholder.png) <!-- A real screenshot can be added later -->

## Features

*   **Classic Snake Mechanics:** Grow by eating food, avoid collisions with walls and self.
*   **AI Integration:** A snake agent that can be trained using the DQN algorithm.
*   **Different Game Modes:**
    *   **Train (Render Off):** Quickly train the agent without a visual interface.
    *   **Play AI:** Watch the trained model's performance.
    *   **Play Manually:** Play the classic snake game using the keyboard.
*   **Model Saving/Loading:** Training progress is saved as a `.zip` file, allowing training to resume later.
*   **Configurable Parameters:** Game grid size (`GRID_SIZE`) and training timesteps (`TIMESTEPS`) can be easily modified in the code.
*   **Pause Feature:** In visual modes (Play AI, Play Manually), the game can be paused with 'P' and quit with 'Q'.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url> # If it's a git repository
    cd <project-folder>
    ```
    *If downloaded as a zip, extract it and navigate into the folder.*

2.  **Install Dependencies:**
    *   Python 3.x must be installed.
    *   Using a virtual environment is recommended:
        ```bash
        python -m venv venv
        source venv/bin/activate  # Linux/macOS
        # venv\Scripts\activate  # Windows
        ```
    *   Install the required packages from `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```
        *(The `requirements.txt` typically includes: `pygame`, `stable-baselines3[extra]`, `gymnasium`)*

## Usage

Navigate to the project's root directory in your terminal and run:

```bash
python snake_game.py
```

This command will display a text-based menu with the following options:

1.  **Train (Render Off, Fast):** Starts training the AI agent. Progress is shown in the terminal. Press `Ctrl+C` to stop training and save the model.
2.  **Play AI (Render On):** Loads the pre-trained `dqn_snake_model.zip` model and shows the AI playing the game. Close the window or press 'Q' to quit.
3.  **Play Manually (Render On):** Play the game yourself using the arrow keys. Close the window or press 'Q' to quit.
4.  **Exit:** Terminates the program.

**Note:** You need to run the training mode (Option 1) at least once to create the `dqn_snake_model.zip` file before using the "Play AI" mode.

## Code Structure

*   `snake_game.py`: Contains the main game logic, the AI environment (`SnakeEnv`), training (`train`), and gameplay (`play_ai`, `play_manual`) functions, along with the menu.
*   `dqn_snake_model.zip`: The saved file for the trained DQN model (created after training).
*   `requirements.txt`: Required Python libraries.
*   `README.md`: This file.
*   `PRD.md`: Project requirements and detailed notes (for development tracking, currently in Turkish).

## Future Development Ideas

*   Experiment with and compare different RL algorithms (PPO, A2C).
*   Integrate TensorBoard for visualizing training metrics.
*   Perform hyperparameter optimization.
*   Add obstacles to the game environment.
*   Visualize the agent's state representation.

## Contributing

(If the project is open source) Contributions are welcome! Please open an issue or submit a pull request. 