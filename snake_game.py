import pygame
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 15}

    def __init__(self, grid_size=40, render_mode=None):
        super(SnakeEnv, self).__init__()

        self.grid_size = grid_size
        self.block_size = 20
        self.width = self.grid_size * self.block_size
        self.height = self.grid_size * self.block_size

        self.action_space = spaces.Discrete(4)

        # Obs: [food_rel_x, food_rel_y, danger_left, danger_front, danger_right, dir_x, dir_y]
        low = np.array([-self.grid_size, -self.grid_size, 0, 0, 0, -1, -1], dtype=np.float32)
        high = np.array([self.grid_size, self.grid_size, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.WHITE = (255, 255, 255)
        self.RED = (213, 50, 80)
        self.GREEN = (0, 255, 0)
        self.BLUE = (50, 153, 213)
        self.BLACK = (0, 0, 0)
        self.HEAD_COLOR = (0, 150, 0)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.paused = False
        self.episode_step = 0

        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption('AI Snake Game')
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def _get_obs(self):
        head_x, head_y = self.snake_body[0]
        food_x, food_y = self.food_pos

        food_rel_x = food_x - head_x
        food_rel_y = food_y - head_y

        dx, dy = 0, 0
        if self.direction == 0: dy = -1
        elif self.direction == 1: dy = 1
        elif self.direction == 2: dx = -1
        elif self.direction == 3: dx = 1

        # Relative danger checks
        current_dir_vector = np.array([dx, dy])
        left_dir_vector = np.array([-dy, dx])
        right_dir_vector = np.array([dy, -dx])

        point_l = (head_x + left_dir_vector[0] * self.block_size, head_y + left_dir_vector[1] * self.block_size)
        point_r = (head_x + right_dir_vector[0] * self.block_size, head_y + right_dir_vector[1] * self.block_size)
        point_f = (head_x + current_dir_vector[0] * self.block_size, head_y + current_dir_vector[1] * self.block_size)

        danger_left = 1.0 if self._is_collision(point_l) else 0.0
        danger_front = 1.0 if self._is_collision(point_f) else 0.0
        danger_right = 1.0 if self._is_collision(point_r) else 0.0

        obs = np.array([
            food_rel_x / self.block_size, # Normalize?
            food_rel_y / self.block_size,
            danger_left,
            danger_front,
            danger_right,
            float(dx),
            float(dy)
        ], dtype=np.float32)
        return obs

    def _get_info(self):
        return {"score": self.score, "snake_length": len(self.snake_body)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        center_x = (self.grid_size // 2) * self.block_size
        center_y = (self.grid_size // 2) * self.block_size
        self.snake_body = [[center_x, center_y],
                           [center_x - self.block_size, center_y],
                           [center_x - 2 * self.block_size, center_y]]
        self.direction = 3 # RIGHT

        self.score = 0
        self.episode_step = 0
        self._place_food()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _place_food(self):
        while True:
            self.food_pos = [random.randrange(0, self.grid_size) * self.block_size,
                             random.randrange(0, self.grid_size) * self.block_size]
            if self.food_pos not in self.snake_body:
                break

    def step(self, action):
        # Prevent snake from reversing
        current_action = self.direction
        if action == 0 and current_action == 1: action = current_action
        elif action == 1 and current_action == 0: action = current_action
        elif action == 2 and current_action == 3: action = current_action
        elif action == 3 and current_action == 2: action = current_action
        self.direction = action

        prev_head = self.snake_body[0][:] # Copy!
        prev_dist_to_food = np.linalg.norm(np.array(prev_head) - np.array(self.food_pos))

        head_x, head_y = self.snake_body[0]
        if self.direction == 0: head_y -= self.block_size
        elif self.direction == 1: head_y += self.block_size
        elif self.direction == 2: head_x -= self.block_size
        elif self.direction == 3: head_x += self.block_size
        new_head = [head_x, head_y]

        terminated = self._is_collision(new_head)
        truncated = False # No time limit yet

        reward = 0
        if terminated:
            reward = -100 # Collision penalty
        else:
            self.snake_body.insert(0, new_head)
            if new_head == self.food_pos:
                self.score += 10
                reward = 50 # Food reward
                self._place_food()
            else:
                self.snake_body.pop()
                new_dist_to_food = np.linalg.norm(np.array(new_head) - np.array(self.food_pos))
                reward += (prev_dist_to_food - new_dist_to_food) * 0.1 # Reward getting closer (tunable)
                reward -= 0.1 # Time penalty (tunable)

        self.episode_step += 1
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _is_collision(self, point=None):
        if point is None: point = self.snake_body[0]
        px, py = point
        # Walls
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return True
        # Self
        if point in self.snake_body[1:]:
            return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(to_rgb_array=True)
        elif self.render_mode == "human":
             self._render_frame()

    def _render_frame(self, to_rgb_array=False):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption('AI Snake Game')
            self.screen = pygame.display.set_mode((self.width, self.height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # -- Human Mode Event Handling --
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                     self.close()
                     print("Window closed, exiting...")
                     sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                        if self.paused: print("Game Paused. P: Resume, Q: Quit")
                        else: print("Game Resumed.")

            while self.paused:
                pause_font = pygame.font.SysFont('arial', 50)
                pause_text = 'PAUSED (P: Resume, Q: Quit)'
                pause_surface = pause_font.render(pause_text, True, self.RED)
                pause_rect = pause_surface.get_rect(center=(self.width / 2, self.height / 2))
                temp_surface = self.screen.copy()
                temp_surface.blit(pause_surface, pause_rect)
                self.screen.blit(temp_surface, (0,0))
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Window closed, exiting...")
                        self.close()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            self.paused = False
                            print("Game Resumed.")
                            break
                        elif event.key == pygame.K_q:
                            print("Q pressed, exiting...")
                            self.close()
                            sys.exit()
                self.clock.tick(15) # Lower FPS during pause
        # -- End Human Mode Event Handling --

        # --- Main Drawing --- #
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill(self.BLACK)

        pygame.draw.rect(canvas, self.WHITE, pygame.Rect(self.food_pos[0], self.food_pos[1], self.block_size, self.block_size))

        head = True
        for pos in self.snake_body:
            color = self.HEAD_COLOR if head else self.GREEN
            pygame.draw.rect(canvas, color, pygame.Rect(pos[0], pos[1], self.block_size, self.block_size))
            pygame.draw.rect(canvas, self.BLACK, pygame.Rect(pos[0], pos[1], self.block_size, self.block_size), 1) # Outline
            head = False

        font = pygame.font.SysFont('arial', 20)
        score_surface = font.render('Score: ' + str(self.score), True, self.WHITE)
        score_rect = score_surface.get_rect(topleft = (10, 10))
        canvas.blit(score_surface, score_rect)

        step_surface = font.render(f'Step: {self.episode_step}', True, self.WHITE)
        step_rect = step_surface.get_rect(topright = (self.width - 10, 10))
        canvas.blit(step_surface, step_rect)
        # --- End Main Drawing --- #

        if self.render_mode == "human":
            self.screen.blit(canvas, canvas.get_rect())
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        elif to_rgb_array:
             # Return np array for SB3
             return np.transpose(
                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
             )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

# --- RL Training --- #
def train(env, total_timesteps=500000, model_path="dqn_snake_model.zip"):
    """Train or continue training the DQN model."""
    # DQN Hyperparameters - tuning these is key
    learning_rate = 0.001
    buffer_size = 10000
    learning_starts = 1000
    batch_size = 64
    tau = 1.0
    gamma = 0.99
    train_freq = 4
    gradient_steps = 1
    target_update_interval = 1000
    exploration_fraction = 0.2
    exploration_initial_eps = 1.0
    exploration_final_eps = 0.05

    model_file_path = os.path.abspath(model_path)
    print(f"Model file: {model_file_path}")

    if os.path.exists(model_file_path):
        print(f"Loading existing model...")
        try:
            model = DQN.load(model_file_path, env=env)
            print("Model loaded. Continuing training.")
            # Note: Replay buffer usually not saved by SB3
            print("(Replay buffer not loaded, agent will collect new data)")
            reset_num_timesteps = False
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead...")
            model = DQN('MlpPolicy', env, learning_rate=learning_rate, buffer_size=buffer_size,
                        learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
                        train_freq=train_freq, gradient_steps=gradient_steps, target_update_interval=target_update_interval,
                        exploration_fraction=exploration_fraction, exploration_initial_eps=exploration_initial_eps,
                        exploration_final_eps=exploration_final_eps, verbose=1)
            reset_num_timesteps = True
    else:
        print(f"Model not found. Creating new model...")
        model = DQN('MlpPolicy', env, learning_rate=learning_rate, buffer_size=buffer_size,
                    learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma,
                    train_freq=train_freq, gradient_steps=gradient_steps, target_update_interval=target_update_interval,
                    exploration_fraction=exploration_fraction, exploration_initial_eps=exploration_initial_eps,
                    exploration_final_eps=exploration_final_eps, verbose=1)
        reset_num_timesteps = True

    print(f"Training for {total_timesteps} timesteps... (Press Ctrl+C to stop)")
    try:
        model.learn(total_timesteps=total_timesteps,
                    log_interval=10, # Log every 10 episodes
                    reset_num_timesteps=reset_num_timesteps)
    except KeyboardInterrupt:
        print("\nTraining interrupted (Ctrl+C).")
    finally:
        # Save model even if interrupted
        print(f"Saving model -> {model_file_path}")
        model.save(model_file_path)
        print("Model saved.")

# --- AI Gameplay --- #
def play_ai(env, model_path="dqn_snake_model.zip"):
    """Load and play the trained model."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found -> {model_path}")
        print("Train the model first (Menu: 1)")
        return

    print(f"Loading model: {model_path}")
    try:
        model = DQN.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("AI Playing... (Press Q or Close Window to Quit)")

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    episodes = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if env.render_mode == 'human' and not pygame.display.get_init():
            print("Game window closed.")
            break

        if terminated or truncated:
            episodes += 1
            print(f"Episode {episodes} finished. Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            if env.render_mode == 'human' and not pygame.display.get_init(): break
            obs, info = env.reset()

    print("AI play mode finished.")

# --- Manual Gameplay --- #
def play_manual(env):
    """Manual play using keyboard."""
    print("Manual Play Mode.")
    print("Controls: Arrow Keys. Quit: Q / Close Window.")
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated:
        action = env.direction

        if env.render_mode == 'human':
            action_taken_by_user = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True; break
                if event.type == pygame.KEYDOWN:
                    action_taken_by_user = True
                    if event.key == pygame.K_UP: action = 0
                    elif event.key == pygame.K_DOWN: action = 1
                    elif event.key == pygame.K_LEFT: action = 2
                    elif event.key == pygame.K_RIGHT: action = 3
                    elif event.key == pygame.K_q: terminated = True; break
            if terminated: break
        else:
            print("Error: Manual play requires render_mode='human'.")
            break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if terminated or truncated:
            print(f"Game Over! Steps: {step_count}, Score: {info['score']}, Total Reward: {total_reward:.2f}")

    print("Manual play mode finished.")

# --- Menu Function --- #
def draw_menu():
    """Simple text menu in console."""
    print("\n--- Snake Game Menu ---")
    print("1. Train (Render Off, Fast)")
    print("2. Play AI (Render On)")
    print("3. Play Manually (Render On)")
    print("4. Exit")
    print("-----------------------")

    while True:
        choice = input("Select (1-4): ")
        if choice in ['1', '2', '3', '4']:
            return choice
        else:
            print("Invalid input.")

# --- Main Execution --- #
if __name__ == '__main__':
    # Config
    GRID_SIZE = 40
    TIMESTEPS = 500000
    MODEL_PATH = "dqn_snake_model.zip"

    # Main loop
    while True:
        choice = draw_menu()

        # Handle choice
        if choice == '1': # TRAIN
            print("Starting training (Render Off)...")
            env = None
            try:
                env = SnakeEnv(grid_size=GRID_SIZE, render_mode=None)
                # check_env(env) # Optional check
                train(env, total_timesteps=TIMESTEPS, model_path=MODEL_PATH)
            except Exception as e:
                print(f"Training error: {e}")
            finally:
                if env: env.close()
            print("Training session finished/interrupted.")

        elif choice == '2': # PLAY AI
            print("Starting AI play mode...")
            env = None
            try:
                env = SnakeEnv(grid_size=GRID_SIZE, render_mode='human')
                play_ai(env, model_path=MODEL_PATH)
            except Exception as e:
                 print(f"AI play error: {e}")
            finally:
                 if env: env.close()

        elif choice == '3': # PLAY MANUAL
            print("Starting manual play mode...")
            env = None
            try:
                env = SnakeEnv(grid_size=GRID_SIZE, render_mode='human')
                play_manual(env)
            except Exception as e:
                 print(f"Manual play error: {e}")
            finally:
                if env: env.close()

        elif choice == '4': # EXIT
            print("Exiting...")
            break

    print("Program finished.") 