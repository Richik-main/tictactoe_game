import numpy as np
import random
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_winner = None
        self.current_player = 1
        return self.board

    def available_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, action):
        i, j = action
        if self.board[i, j] == 0:
            self.board[i, j] = self.current_player
            if self.check_winner(i, j):
                self.current_winner = self.current_player
            self.current_player = -self.current_player
            return True
        return False

    def check_winner(self, x, y):
        row_win = all([self.board[x, j] == self.current_player for j in range(3)])
        col_win = all([self.board[i, y] == self.current_player for i in range(3)])
        diag1_win = all([self.board[i, i] == self.current_player for i in range(3)])
        diag2_win = all([self.board[i, 2-i] == self.current_player for i in range(3)])
        return row_win or col_win or diag1_win or diag2_win

    def is_draw(self):
        return len(self.available_actions()) == 0 and self.current_winner is None

    def game_over(self):
        return self.current_winner is not None or self.is_draw()

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.q_table = {}       # Q-table

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state.flatten()), action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        best_next_q = 0
        if not game.game_over():
            best_next_q = max([self.get_q_value(next_state, a) for a in game.available_actions()])
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[(tuple(state.flatten()), action)] = new_q

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(game.available_actions())
        else:
            q_values = [self.get_q_value(state, action) for action in game.available_actions()]
            max_q = max(q_values)
            return game.available_actions()[q_values.index(max_q)]

game = TicTacToe()
agent = QLearningAgent()

def train(agent, episodes=10000):
    for _ in range(episodes):
        state = game.reset()
        while not game.game_over():
            action = agent.choose_action(state)
            game.make_move(action)
            reward = 0
            if game.current_winner == 1:
                reward = 1
            elif game.current_winner == -1:
                reward = -1
            elif game.is_draw():
                reward = 0.5
            next_state = game.board.copy()
            agent.update_q_value(state, action, reward, next_state)
            state = next_state

train(agent)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    row = data['row']
    col = data['col']
    action = (row, col)
    
    if action in game.available_actions():
        # Human move
        state = game.board.copy()
        game.make_move(action)
        reward = 0
        if game.game_over():
            if game.current_winner == 1:
                reward = -1  # User win is negative reward for agent
            elif game.is_draw():
                reward = 0.5
        next_state = game.board.copy()
        agent.update_q_value(state, action, reward, next_state)

        if not game.game_over():
            # Agent move
            state = game.board.copy()
            agent_action = agent.choose_action(state)
            game.make_move(agent_action)
            reward = 0
            if game.game_over():
                if game.current_winner == -1:
                    reward = 1  # Agent win is positive reward for agent
                elif game.is_draw():
                    reward = 0.5
            next_state = game.board.copy()
            agent.update_q_value(state, agent_action, reward, next_state)

        response = {
            'board': game.board.tolist(),
            'game_over': game.game_over(),
            'winner': game.current_winner
        }
    else:
        response = {'error': 'Invalid move'}

    return jsonify(response)

@app.route('/reset', methods=['POST'])
def reset():
    game.reset()
    return jsonify({'board': game.board.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
