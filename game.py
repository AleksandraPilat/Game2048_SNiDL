import numpy as np
import torch
from numpy.random import *
import math
import random


def NewCreateTile(list1):
    """
    Adds a new tile (either 2 or 4) to a random empty position on the board.
    """
    num = random.choice([2] * 9 + [4])  # Choose either 2 or 4
    while True:
        x = random.randint(0, 3)  # Random row index
        y = random.randint(0, 3)  # Random column index
        if list1[x][y] == 0:  # If the position is empty
            list1[x][y] = num  # Place the chosen number
            break
    return list1


def MoveTile2(list1):
    """
    Simulates the movement and merging of tiles in one row or column.
    """
    list2 = [0] * 4  # Initialize the new row/column
    score = 0  # Initialize the score
    isMoved = False  # Flag to check if any movement happened
    notMerged = False  # Flag to track if merging has occurred
    idxTgt = 0  # Target index for merging

    for idxSrc in range(4):
        if list1[idxSrc] == 0:
            pass  # Skip empty spaces
        elif notMerged:
            if list1[idxSrc] == list2[idxTgt]:
                # Merge tiles if they are equal
                list2[idxTgt] += list1[idxSrc]
                score += list2[idxTgt]
                notMerged = False
                isMoved = True
                idxTgt += 1
            else:
                idxTgt += 1
                list2[idxTgt] = list1[idxSrc]
                notMerged = True
                isMoved = isMoved or idxTgt != idxSrc
        else:
            list2[idxTgt] = list1[idxSrc]
            notMerged = True
            isMoved = isMoved or idxTgt != idxSrc

    return (score, isMoved, list2)


def newBoard():
    """
    Creates and returns a new 4x4 board filled with zeros.
    """
    return [[0] * 4 for _ in range(4)]


def CloneBoard(list1):
    """
    Creates and returns a copy of the given board.
    """
    return [[list1[i][j] for j in range(4)] for i in range(4)]


def DoUp(list1):
    """
    Simulates moving the tiles up and merging them.
    """
    temp_B = newBoard()  # Initialize a temporary board
    temp_score = [0] * 4  # Initialize score for each column
    isMoved = [False] * 4  # Track movement for each column

    for i in range(4):
        # Perform tile movement and merging for each column
        (temp_score[i], isMoved[i], [temp_B[0][i], temp_B[1][i], temp_B[2][i], temp_B[3][i]]) \
            = MoveTile2([list1[0][i], list1[1][i], list1[2][i], list1[3][i]])

    return (temp_B, sum(temp_score), any(isMoved))


def DoRight(list1):
    """
    Simulates moving the tiles to the right and merging them.
    """
    temp_B = newBoard()
    temp_score = [0] * 4
    isMoved = [False] * 4

    for i in range(4):
        # Perform tile movement and merging for each row (right direction)
        (temp_score[i], isMoved[i], [temp_B[i][3], temp_B[i][2], temp_B[i][1], temp_B[i][0]]) \
            = MoveTile2([list1[i][3], list1[i][2], list1[i][1], list1[i][0]])

    return (temp_B, sum(temp_score), any(isMoved))


def DoDown(list1):
    """
    Simulates moving the tiles down and merging them.
    """
    temp_B = newBoard()
    temp_score = [0] * 4
    isMoved = [False] * 4

    for i in range(4):
        # Perform tile movement and merging for each column (down direction)
        (temp_score[i], isMoved[i], [temp_B[3][i], temp_B[2][i], temp_B[1][i], temp_B[0][i]]) \
            = MoveTile2([list1[3][i], list1[2][i], list1[1][i], list1[0][i]])

    return (temp_B, sum(temp_score), any(isMoved))


def DoLeft(list1):
    """
    Simulates moving the tiles to the left and merging them.
    """
    temp_B = newBoard()
    temp_score = [0] * 4
    isMoved = [False] * 4

    for i in range(4):
        # Perform tile movement and merging for each row (left direction)
        (temp_score[i], isMoved[i], [temp_B[i][0], temp_B[i][1], temp_B[i][2], temp_B[i][3]]) \
            = MoveTile2([list1[i][0], list1[i][1], list1[i][2], list1[i][3]])

    return (temp_B, sum(temp_score), any(isMoved))


actions = [DoUp, DoRight, DoDown, DoLeft]


def GameOverCheck(list1):
    """
    Checks if the game is over.
    """
    for act in actions:
        _, _, isMoved = act(list1)
        if isMoved:
            return False
    return True


def BoardDisply(list1):
    """
    Displays the board.
    """
    for i in range(4):
        for j in range(4):
            print("{0:6}".format(list1[i][j]), end="")
        print("")


def Board8(image, model, device):
    """
    Converts the board image into 8 different transformations and evaluates them using a model.
    """
    tempB = np.zeros([4, 4])

    # Convert board values to log2 format
    for i in range(16):
        t_num = 0
        if image[i // 4][i % 4] != 0:
            t_num = int(math.log2(image[i // 4][i % 4]))
        tempB[i // 4][i % 4] = t_num

    # Generate different transformations of the board
    g0 = tempB
    g1 = np.flip(g0, axis=0)
    g2 = np.flip(g0, axis=1)
    g3 = np.flip(g2, axis=0)
    r0 = np.transpose(g0)
    r1 = np.flip(r0, axis=0)
    r2 = np.flip(r0, axis=1)
    r3 = np.flip(r2, axis=0)

    # Create input tensor for the model
    inputB = np.zeros([8, 4, 4, 16])
    gcount = 0
    for g in [g0, r2, g3, r1, g2, r0, g1, r3]:
        for i in range(16):
            inputB[gcount][i // 4][i % 4][int(g[i // 4][i % 4])] = 1
        gcount += 1

    inputB = torch.tensor(inputB, dtype=torch.float32).to(device)
    inputB = inputB.permute(0, 3, 1, 2)

    with torch.no_grad():
        prev = model(inputB).cpu().numpy()

    # Aggregate predictions from different transformations
    P = np.zeros([4], dtype=np.float32)
    Pcount = np.zeros([4])

    B0(prev[0][:], P, Pcount)
    B1(prev[1][:], P, Pcount)
    B2(prev[2][:], P, Pcount)
    B3(prev[3][:], P, Pcount)
    B4(prev[4][:], P, Pcount)
    B5(prev[5][:], P, Pcount)
    B6(prev[6][:], P, Pcount)
    B7(prev[7][:], P, Pcount)

    return P, Pcount


# Helper functions to aggregate predictions from different transformations
def B0(pre, P, Pcount):  # 0123
    for i in range(4):
        P[i] += pre[i]
    Pcount[np.argmax(pre)] += 1


def B1(pre, P, Pcount):  # 12840
    tempP = np.zeros([4])
    tempP[0] = pre[1]
    tempP[1] = pre[2]
    tempP[2] = pre[3]
    tempP[3] = pre[0]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B2(pre, P, Pcount):  # 15141312
    tempP = np.zeros([4])
    tempP[0] = pre[2]
    tempP[1] = pre[3]
    tempP[2] = pre[0]
    tempP[3] = pre[1]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B3(pre, P, Pcount):  # 371115
    tempP = np.zeros([4])
    tempP[0] = pre[3]
    tempP[1] = pre[0]
    tempP[2] = pre[1]
    tempP[3] = pre[2]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B4(pre, P, Pcount):  # 3210
    tempP = np.zeros([4])
    tempP[0] = pre[0]
    tempP[1] = pre[3]
    tempP[2] = pre[2]
    tempP[3] = pre[1]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B5(pre, P, Pcount):  # 04812
    tempP = np.zeros([4])
    tempP[0] = pre[3]
    tempP[1] = pre[2]
    tempP[2] = pre[1]
    tempP[3] = pre[0]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B6(pre, P, Pcount):  # 12131415
    tempP = np.zeros([4])
    tempP[0] = pre[2]
    tempP[1] = pre[1]
    tempP[2] = pre[0]
    tempP[3] = pre[3]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def B7(pre, P, Pcount):  # 151173
    tempP = np.zeros([4])
    tempP[0] = pre[1]
    tempP[1] = pre[0]
    tempP[2] = pre[3]
    tempP[3] = pre[2]
    Pcount[np.argmax(tempP)] += 1
    for i in range(4):
        P[i] += tempP[i]


def run_game_simulation(model, model_name, device):
    """
    Run the game simulation using the given model.
    """
    num_trial = 1000  # Number of trials
    sum_score = hi_score = clear_count = 0

    for play in range(num_trial):
        current_score = count = 0  # Reset score and move count
        current_list = NewCreateTile(NewCreateTile(newBoard()))

        while True:  # Play one game
            if GameOverCheck(current_list):
                break  # Check for game over

            count += 1  # Increment move count

            # Copy the board and get model predictions
            image = CloneBoard(current_list)
            prev, Pcount = Board8(image, model, device)

            while True:
                select = -1
                pmax = -1
                # Select the best move based on probabilities
                for i in range(4):
                    if pmax < Pcount[i]:
                        pmax = Pcount[i]
                        select = i
                    elif pmax == Pcount[i] and prev[select] < prev[i]:
                        pmax = Pcount[i]
                        select = i
                Pcount[select] = -1
                action = actions[select]
                _, _, isMoved = action(CloneBoard(current_list))
                if isMoved:
                    break

            # Perform the selected action
            current_list, temp_score, _ = action(current_list)
            current_score += temp_score
            current_list = NewCreateTile(current_list)

        # Update scores
        if hi_score <= current_score:
            hi_score = current_score
        sum_score += current_score
        image = CloneBoard(current_list)

        max_tile = 0
        for i in range(4):
            for j in range(4):
                if max_tile < image[i][j]:
                    max_tile = image[i][j]

        if 2048 <= max_tile:
            clear_count += 1

        # Write results to file
        with open(f"result_{model_name}.txt", "a") as f:
            moji = f"{play + 1},{current_score},{max_tile},{count},{clear_count}\n"
            f.write(moji)

    # Write summary statistics to file
    with open(f"result_{model_name}.txt", "a") as f:
        moji = f"average score is: {sum_score / num_trial}\n"
        moji += f"best score is: {hi_score}\n"
        moji += f"clear: {clear_count / 10}\n"
        f.write(moji)
