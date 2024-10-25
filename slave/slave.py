import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import plotly.graph_objs as go
import plotly.io as pio
from tqdm import tqdm
import webbrowser
import os
from torch.utils.tensorboard import SummaryWriter  # TensorBoard Integration
import pickle  # For saving replay buffer and other objects
import pygame  # For visualization
import time  # For timing episodes

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Define a transition tuple for the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class SlaveCardGameEnv:
    """
    Environment for the Slave card game.
    Two players: Player 1 and Player 2 (both AI agents).
    Players take turns to play a higher card or pass.
    The first to empty their hand wins.
    """
    def __init__(self, visualize=False):
        self.action_space = 53  # 0-51 for cards, 52 for pass
        self.observation_space = 52 * 3 + 1  # Player hand, opponent hand, last move, current player
        self.ranks = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
        self.suits = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
        self.rank_to_index = {rank: idx for idx, rank in enumerate(self.ranks)}
        self.suit_to_index = {suit: idx for idx, suit in enumerate(self.suits)}
        self.visualize = visualize
        
        # Initialize Pygame if visualization is enabled
        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption('Slave Card Game')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)  # Initialize font before calling load_card_images
            self.card_images = self.load_card_images()  # Load card images after initializing font

        self.reset()

    def reset(self):
        """
        Resets the game to initial state.
        """
        self.deck = list(range(52))  # Represent cards as indices 0-51
        random.shuffle(self.deck)
        self.player_hand = set(self.deck[:13])
        self.opponent_hand = set(self.deck[13:26])
        self.table_cards = []
        self.last_move = []
        self.current_player = 0  # 0: Player 1, 1: Player 2
        self.done = False
        self.winner = None
        return self._get_obs()

    def _get_obs(self):
        """
        Constructs the observation.
        """
        player_hand_obs = self.cards_to_obs(self.player_hand)
        opponent_hand_obs = self.cards_to_obs(self.opponent_hand)
        last_move_obs = self.cards_to_obs(self.last_move)
        current_player_obs = np.array([self.current_player], dtype=np.float32)
        obs = np.concatenate([player_hand_obs, opponent_hand_obs, last_move_obs, current_player_obs])
        return obs

    def cards_to_obs(self, cards):
        """
        Converts a set or list of card indices to a binary vector.
        """
        obs = np.zeros(52, dtype=np.float32)
        if isinstance(cards, set):
            for card in cards:
                obs[card] = 1.0
        elif isinstance(cards, list):
            for card in cards:
                obs[card] = 1.0
        return obs

    def is_valid_move(self, action, player_hand):
        """
        Checks if the action is valid.
        """
        if action == 52:
            # Pass is valid unless no previous move
            return len(self.last_move) > 0
        if action not in player_hand:
            return False
        if not self.last_move:
            return True  # Any card can be played if no previous move
        # Compare ranks
        new_rank = action // 4
        last_rank = self.last_move[0] // 4
        return new_rank > last_rank

    def step(self, action):
        """
        Executes the action and updates the game state.
        Returns next_state, reward, done.
        """
        if self.done:
            raise ValueError("Game has ended. Please reset the environment.")

        reward = 0

        # Determine current player's hand
        if self.current_player == 0:
            player_hand = self.player_hand
        else:
            player_hand = self.opponent_hand

        valid = self.is_valid_move(action, player_hand)

        if action == 52:
            if self.last_move:
                # Pass and switch player
                self.current_player = 1 - self.current_player
                reward = 0  # No reward for passing
        else:
            if valid:
                player_hand.remove(action)
                self.last_move = [action]
                self.current_player = 1 - self.current_player  # Switch to other player
                reward = 1  # Reward for playing a valid card
                if not player_hand:
                    self.done = True
                    self.winner = f'Player {2 - self.current_player}'
                    reward = 100  # Big reward for winning
            else:
                reward = -1  # Punishment for invalid move

        next_state = self._get_obs()
        if self.visualize:
            self.render()
        return next_state, reward, self.done

    def render(self):
        """
        Renders the current game state using Pygame.
        """
        if not self.visualize:
            return

        self.screen.fill((0, 128, 0))  # Green background for table

        # Display player hands and last move
        self.display_hand(self.player_hand, position=(50, 450), label='Player 1 Hand')
        self.display_hand(self.opponent_hand, position=(50, 50), label='Player 2 Hand', hidden=False)
        self.display_last_move()
        self.display_current_player()

        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def load_card_images(self):
        """
        Loads card images from files.
        """
        card_images = {}
        suits_symbols = {'Clubs': 'C', 'Diamonds': 'D', 'Hearts': 'H', 'Spades': 'S'}
        ranks_symbols = {'10': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A', '2': '2'}
        for i in range(3, 10):
            ranks_symbols[str(i)] = str(i)
        for rank in self.ranks:
            for suit in self.suits:
                card_name = ranks_symbols[rank] + suits_symbols[suit]
                image = pygame.Surface((50, 70))
                image.fill((255, 255, 255))
                pygame.draw.rect(image, (0, 0, 0), image.get_rect(), 2)
                text = self.font.render(f'{rank[0]}{suits_symbols[suit]}', True, (0, 0, 0))  # Using self.font here
                image.blit(text, (5, 25))
                index = self.ranks.index(rank) * 4 + self.suits.index(suit)
                card_images[index] = image
        # Back of card
        back_image = pygame.Surface((50, 70))
        back_image.fill((0, 0, 255))
        pygame.draw.rect(back_image, (0, 0, 0), back_image.get_rect(), 2)
        card_images['back'] = back_image
        return card_images

    def display_hand(self, hand, position, label, hidden=False):
        """
        Displays a hand of cards at the given position.
        """
        x, y = position
        label_surface = self.font.render(label, True, (255, 255, 255))
        self.screen.blit(label_surface, (x, y - 30))
        for idx, card in enumerate(sorted(hand)):
            if hidden:
                image = self.card_images['back']
            else:
                image = self.card_images[card]
            self.screen.blit(image, (x + idx * 55, y))

    def display_last_move(self):
        """
        Displays the last move on the table.
        """
        label_surface = self.font.render('Last Move', True, (255, 255, 255))
        self.screen.blit(label_surface, (350, 250))
        if self.last_move:
            card = self.last_move[0]
            image = self.card_images[card]
            self.screen.blit(image, (375, 280))
        else:
            no_move_surface = self.font.render('None', True, (255, 255, 255))
            self.screen.blit(no_move_surface, (375, 280))

    def display_current_player(self):
        """
        Displays which player's turn it is.
        """
        player = f"Player {self.current_player + 1}'s Turn"
        player_surface = self.font.render(player, True, (255, 255, 0))
        self.screen.blit(player_surface, (320, 550))

    def index_to_card(self, index):
        """
        Maps an index back to a card string.
        """
        rank = self.ranks[index // 4]
        suit = self.suits[index % 4]
        return f"{rank} of {suit}"

class DQN(nn.Module):
    """
    Deep Q-Network model.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Replay Buffer for storing transitions.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """
        Saves a transition.
        """
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        """
        Samples a batch of transitions.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def select_action(model: nn.Module, state: np.ndarray, epsilon: float, action_space: int, device: torch.device, valid_actions: list) -> int:
    """
    Selects an action using epsilon-greedy policy.
    """
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            q_values = q_values.cpu().numpy()[0]
            # Mask invalid actions
            invalid_actions = set(range(action_space)) - set(valid_actions)
            for action in invalid_actions:
                q_values[action] = -np.inf
            return int(np.argmax(q_values))

def train_dqn(model: nn.Module, target_model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
             replay_buffer: ReplayBuffer, batch_size: int, gamma: float, device: torch.device) -> float:
    """
    Trains the DQN model using a batch of transitions from the replay buffer.
    Returns the loss value.
    """
    if len(replay_buffer) < batch_size:
        return None  # Not enough data to train

    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Convert to tensors
    state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
    next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)

    # Current Q values
    current_q = model(state_batch).gather(1, action_batch)

    # Compute target Q values using Double DQN
    with torch.no_grad():
        next_q_main = model(next_state_batch)
        next_actions = next_q_main.argmax(1).unsqueeze(1)
        next_q_target = target_model(next_state_batch).gather(1, next_actions)
        target_q = reward_batch + gamma * next_q_target * (1 - done_batch)

    # Compute loss
    loss = criterion(current_q, target_q)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def update_plotly_figure(values: list, episodes: list, filename: str = 'training_reward.html', refresh_time: int = 5):
    """
    Creates and saves a Plotly figure with the given values and episodes.
    Injects a meta-refresh tag to auto-reload the page.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=episodes, y=values, mode='lines+markers', name='Total Reward'))
    fig.update_layout(title='Total Reward per Episode',
                      xaxis_title='Episode',
                      yaxis_title='Total Reward',
                      template='plotly_dark')  # Optional: dark theme

    # Generate the Plotly figure div
    fig_div = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

    # Define the HTML template with meta-refresh
    html_template = f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="{refresh_time}">
    </head>
    <body>
        {fig_div}
    </body>
    </html>
    """

    # Save the HTML to file
    with open(filename, 'w') as f:
        f.write(html_template)

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, replay_buffer: ReplayBuffer,
                   episode: int, epsilon: float, reward_history: list, filename: str):
    """
    Saves the training checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer': list(replay_buffer.buffer),  # Convert deque to list for pickling
        'episode': episode,
        'epsilon': epsilon,
        'reward_history': reward_history
    }
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at episode {episode} as '{filename}'.")

def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, replay_buffer: ReplayBuffer,
                   filename: str, device: torch.device):
    """
    Loads the training checkpoint.
    """
    with open(filename, 'rb') as f:
        checkpoint = pickle.load(f)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    replay_buffer.buffer = deque(checkpoint['replay_buffer'], maxlen=replay_buffer.buffer.maxlen)
    episode = checkpoint['episode']
    epsilon = checkpoint['epsilon']
    reward_history = checkpoint['reward_history']
    print(f"Checkpoint loaded from '{filename}' at episode {episode}.")
    return episode, epsilon, reward_history

def main():
    # Hyperparameters
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 700
    BATCH_SIZE = 64
    GAMMA = 0.99
    LEARNING_RATE = 1e-3
    REPLAY_BUFFER_CAPACITY = 10000
    TARGET_UPDATE_FREQ = 700
    EPSILON_START = 1.0
    EPSILON_DECAY = 0.999  # Decay epsilon faster so the agent explores less over time
    EPSILON_END = 0.01  # Reduce the final epsilon to explore even less
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VISUALIZE = True  # Set to True to visualize with Pygame

    # Initialize environment, model, target model, optimizer, criterion, replay buffer
    env = SlaveCardGameEnv(visualize=VISUALIZE)
    model = DQN(env.observation_space, env.action_space).to(DEVICE)
    target_model = DQN(env.observation_space, env.action_space).to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    # TensorBoard Initialization
    writer = SummaryWriter('runs/slave_card_game')

    # Initialize Plotly figure
    plotly_filename = 'training_reward.html'
    update_plotly_figure([], [], filename=plotly_filename, refresh_time=5)

    # Open the Plotly HTML file in the default web browser
    filepath = os.path.abspath(plotly_filename)
    if not os.path.exists(plotly_filename):
        webbrowser.open(f'file://{filepath}')

    # Initialize training variables
    epsilon = EPSILON_START
    reward_history = []
    episode_indices = []
    steps_done = 0

    # Define checkpoint filename
    checkpoint_filename = 'dqn_slave_card_game_checkpoint.pkl'

    # Check if a checkpoint exists to resume training
    if os.path.exists(checkpoint_filename):
        user_input = input(f"A checkpoint '{checkpoint_filename}' was found. Do you want to resume training from it? (y/n): ").strip().lower()
        if user_input == 'y':
            try:
                episode, epsilon, reward_history = load_checkpoint(model, optimizer, replay_buffer, checkpoint_filename, DEVICE)
                print(f"Resuming training from episode {episode} with epsilon {epsilon:.4f}.")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting training from scratch.")
                episode = 0
        else:
            print("Starting training from scratch.")
            episode = 0
    else:
        print("No checkpoint found. Starting training from scratch.")
        episode = 0

    print("Starting training... Press Ctrl+C to interrupt and save progress.")

    try:
        # Use tqdm for progress bar
        for episode in tqdm(range(episode + 1, NUM_EPISODES + 1), desc="Training Episodes", initial=episode):
            state = env.reset()
            done = False
            step = 0
            total_reward = 0  # Accumulate total reward per episode
            while not done and step < MAX_STEPS_PER_EPISODE:
                step += 1
                # Determine current player's hand
                if env.current_player == 0:
                    player_hand = env.player_hand
                else:
                    player_hand = env.opponent_hand

                # Get valid actions
                if not env.last_move:
                    valid_actions = list(player_hand) + [52]  # All cards in hand + Pass
                else:
                    last_rank = env.last_move[0] // 4
                    valid_cards = [a for a in player_hand if (a // 4) > last_rank]
                    if valid_cards:
                        valid_actions = valid_cards + [52]  # Valid cards + Pass
                    else:
                        valid_actions = [52]  # Only Pass

                # Agent selects action
                action = select_action(model, state, epsilon, env.action_space, DEVICE, valid_actions)
                next_state, reward, done = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Train the model
                loss = train_dqn(model, target_model, optimizer, criterion, replay_buffer, BATCH_SIZE, GAMMA, DEVICE)
                if loss is not None:
                    writer.add_scalar('Loss/train', loss, episode)

                # Update target network
                steps_done += 1
                if steps_done % TARGET_UPDATE_FREQ == 0:
                    target_model.load_state_dict(model.state_dict())

                if done:
                    break

            # If the game didn't finish due to a win but because of max steps, apply a punishment
            if step >= MAX_STEPS_PER_EPISODE and not env.done:
                total_reward -= 10  # Apply a punishment for not winning before the episode ends

            # Decay epsilon
            if epsilon > EPSILON_END:
                epsilon *= EPSILON_DECAY
                epsilon = max(EPSILON_END, epsilon)

            reward_history.append(total_reward)
            episode_indices.append(episode)
            update_plotly_figure(reward_history, episode_indices, filename=plotly_filename, refresh_time=5)

            # Logging and TensorBoard
            if episode % 50 == 0:
                avg_reward = np.mean(reward_history[-50:])
                tqdm.write(f"Episode {episode}/{NUM_EPISODES}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
                writer.add_scalar('Epsilon', epsilon, episode)
                writer.add_scalar('Avg Reward', avg_reward, episode)

            # Model Checkpointing every 200 episodes
            if episode % 200 == 0:
                checkpoint_file = f'dqn_slave_card_game_episode_{episode}.pth'
                torch.save(model.state_dict(), checkpoint_file)
                tqdm.write(f"Model saved at episode {episode} as '{checkpoint_file}'.")
                # Save the replay buffer and other variables
                save_checkpoint(model, optimizer, replay_buffer, episode, epsilon, reward_history, checkpoint_filename)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        save_checkpoint(model, optimizer, replay_buffer, episode, epsilon, reward_history, checkpoint_filename)
        print("Checkpoint saved. Exiting training.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Saving checkpoint before exiting...")
        save_checkpoint(model, optimizer, replay_buffer, episode, epsilon, reward_history, checkpoint_filename)
        print("Checkpoint saved. Exiting training.")
    else:
        print("Training completed without interruptions.")

    finally:
        # Final Plotly update
        update_plotly_figure(reward_history, episode_indices, filename=plotly_filename, refresh_time=5)

        # Save the final trained model if training wasn't interrupted
        if not os.path.exists(checkpoint_filename):
            torch.save(model.state_dict(), 'dqn_slave_card_game_final.pth')
            print("Trained model saved as 'dqn_slave_card_game_final.pth'.")

        # Close TensorBoard writer
        writer.close()

if __name__ == "__main__":
    main()
