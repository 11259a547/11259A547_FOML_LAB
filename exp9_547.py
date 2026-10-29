import numpy as np                                # fast maths
import gymnasium as gym                           # the game world
game = gym.make("FrozenLake-v1", is_slippery=False)  # the 4x4 ice grid
Q = np.zeros((16, 4))                             # score-sheet: 16 squares x 4 moves, all 0
for round in range(2000):                         # play 2000 practice games
    state, _ = game.reset()                       # start at the beginning
    done = False                                  # not finished yet
    while not done:                               # keep moving until the game ends
        if np.random.rand() < 0.2:                # 20% of the time: explore
            move = game.action_space.sample()     # pick a random move
        else:                                     # otherwise: use the best known move
            move = np.random.choice(np.flatnonzero(Q[state] == Q[state].max()))  # FIX: best move for this square
        new_state, reward, done, _, _ = game.step(move)  # make the move, see the result
        Q[state, move] += 0.8 * (reward + 0.95 * np.max(Q[new_state]) - Q[state, move])  # learn: update the score
        state = new_state                         # move to the new square
print("Training done. The agent has learned the path.")  # finished learning

arrows = ["left", "down", "right", "up"]          # names for moves 0, 1, 2, 3
for square in [0, 6, 10, 14]:                     # check a few squares
    best = np.argmax(Q[square])                   # best move for that square
    print("From square", square, "-> best move:", arrows[best])  # show it

    goal = 15  # the goal square number
    wins = 0  # counter for won games
    for game_no in range(100):  # play 100 test games
        state, _ = game.reset()  # start again
        for step in range(100):  # at most 100 moves per game
            state, reward, end, cut, _ = game.step(np.argmax(Q[state]))  # always use the best move
            if end or cut:  # game over?
                break  # stop this game
        if state == goal:  # did we finish on the goal?
            wins += 1  # count a win
    print("Success rate:", wins, "out of 100 games won")  # show the result
