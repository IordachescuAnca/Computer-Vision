import sys
import numpy as np
import pdb


def select_random_path(E):
    # pentru linia 0 alegem primul pixel in mod aleator
    line = 0
    col = np.random.randint(low=0, high=E.shape[1], size=1)[0]
    path = [(line, col)]
    for i in range(E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        # coloana depinde de coloana pixelului anterior
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.random.randint(low=0, high=2, size=1)[0]
        elif path[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.random.randint(low=-1, high=1, size=1)[0]
        else:
            opt = np.random.randint(low=-1, high=2, size=1)[0]
        col = path[-1][1] + opt
        path.append((line, col))

    return path


def select_greedy_path(E):
    line = 0
    col = np.argmin(E[0])
    path = [(line, col)]
    #print(E.shape)
    for i in range(1, E.shape[0]):
        line = i
        previous_col = path[-1][1]

        if previous_col == 0:
            if E[line][previous_col] <= E[line][previous_col+1]:
                col = previous_col
            else:
                col = previous_col + 1
        elif previous_col == E.shape[1] - 1:
            if E[line][previous_col-1] <= E[line][previous_col]:
                col = previous_col-1
            else:
                col = previous_col
        else:
            if E[line][previous_col] <= E[line][previous_col+1]:
                col = previous_col
            else:
                col = previous_col + 1

            vmin = E[line][col]
            if E[line][previous_col-1] <= vmin:
                col = previous_col-1

        path.append((line, col))

    return path


def select_dynamic_programming_path(E):
    M = np.zeros(E.shape)
    M[0, :] = E[0, :]
    for i in range(1, M.shape[0]):
        M[i, 0] = min(M[i - 1, 0], M[i - 1, 1]) + E[i, 0]
        for j in range(1, M.shape[1]):
            if j == M.shape[1]-1:
                M[i, j] = min(M[i-1, j], M[i-1, j-1])
            else:
                M[i, j] = min(M[i-1, j], M[i-1, j+1], M[i-1, j-1])
            M[i,j] += E[i,j]

    line = M.shape[0] - 1
    col = np.argmin(M[M.shape[0]-1])
    path = [(line, col)]

    while line >= 1:
        line -= 1
        if col == 0:
            if M[line][col] <= M[line][col+1]:
                col = col
            else:
                col = col + 1
        elif col == M.shape[1] - 1:
            if M[line][col-1] <= M[line][col]:
                col = col-1
            else:
                col = col
        else:
            if M[line][col] <= M[line][col+1]:
                col = col
            else:
                col = col + 1
            vmin = M[line][col]
            if M[line][col-1] <= vmin:
                col = col-1
        path.append((line, col))

    path.reverse()

    return path

def select_path(E, method):
    if method == 'aleator':
        return select_random_path(E)
    elif method == 'greedy':
        return select_greedy_path(E)
    elif method == 'programareDinamica':
        return select_dynamic_programming_path(E)
    else:
        print('The selected method %s is invalid.' % method)
        sys.exit(-1)