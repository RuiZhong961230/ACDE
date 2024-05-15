import os
from copy import deepcopy
import numpy as np
from sklearn import datasets

import warnings
warnings.filterwarnings("ignore")


PopSize = 30
DimSize = 10
LB = [0] * DimSize
UB = [50] * DimSize
TrialRuns = 30

MaxIter = 100
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0

BestIndi = None
FitBest = float("inf")

def Euclid(X, Y):
    return (X[0] - Y[0]) ** 2 + (X[1] - Y[1]) ** 2


def WSN_fit(indi, radius=5):
    scale = (0, 50)
    radius_pow = radius * radius
    lb, ub = scale[0], scale[1]
    pos = np.array(indi).reshape(-1, 2)
    sum_sensor = 0
    inner = 0
    for i in np.linspace(lb, ub, 32):
        for j in np.linspace(lb, ub, 32):
            sum_sensor += 1
            for particle in pos:
                if Euclid(particle, [i, j]) <= radius_pow:
                    inner += 1
                    break
    return inner / sum_sensor


def Initial():
    global Pop, FitPop, DimSize, BestIndi, FitBest
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = WSN_fit(Pop[i])
    FitBest = max(FitPop)
    BestIndi = deepcopy(Pop[np.argmin(FitPop)])


def RunACDE():
    global Pop, FitPop, LB, UB, PopSize, DimSize, BestIndi, FitBest, curIter, MaxIter

    muF = 0.4 * (1 - curIter / (MaxIter + 1)) + 0.3
    sigmaF = 0.3 * (1 - curIter / (MaxIter + 1)) + 0.1

    Weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    for i in range(PopSize):
        IDX = np.random.randint(0, PopSize)
        while IDX == i:
            IDX = np.random.randint(0, PopSize)
        candi = list(range(0, PopSize))
        candi.remove(i)
        candi.remove(IDX)
        r1, r2 = np.random.choice(candi, 2, replace=False)

        F1 = np.random.normal(muF, sigmaF)
        F2 = np.random.normal(muF, sigmaF)
        if FitPop[IDX] >= FitPop[i]:  # DE/winner-to-opt/1
            Off = Pop[IDX] + F1 * (BestIndi - Pop[IDX]) + F2 * (Pop[r1] - Pop[r2])
        else:
            Off = Pop[i] + F1 * (BestIndi - Pop[i]) + F2 * (Pop[r1] - Pop[r2])

        for j in range(DimSize):
            w = np.random.choice(Weights)
            Off[j] = w * Off[j] + (1 - w) * Pop[i][j] + np.random.uniform(-1, 1)

        Off = np.clip(Off, LB, UB)
        FitOff = WSN_fit(Off)
        if FitOff > FitPop[i]:
            Pop[i] = Off.copy()
            FitPop[i] = FitOff
            if FitOff > FitBest:
                FitBest = FitOff
                BestIndi = Off.copy()


def main_WSN(Dim):
    global Pop, DimSize, LB, UB, FitBest, BestIndi, curIter
    DimSize = Dim
    LB, UB = [0] * Dim, [50] * Dim
    Pop = np.zeros((PopSize, Dim))

    All_Trial_Best = []
    All_Best = []
    for j in range(TrialRuns):
        curIter = 0
        Best_list = []
        np.random.seed(2024 + 88 * j)
        Initial()
        while curIter < MaxIter:
            RunACDE()
            curIter += 1
            Best_list.append(FitBest)

        All_Best.append(BestIndi)
        All_Trial_Best.append(Best_list)
    np.savetxt("./ACDE_Data/WSN/Grid/Obj/WSN_" + str(int(Dim/2)) + ".csv", All_Trial_Best, delimiter=",")
    np.savetxt("./ACDE_Data/WSN/Grid/Sol/WSN_" + str(int(Dim/2)) + ".csv", All_Best, delimiter=",")


if __name__ == "__main__":
    if os.path.exists('./ACDE_Data/WSN/Grid/Obj') == False:
        os.makedirs('./ACDE_Data/WSN/Grid/Obj')
    if os.path.exists('./ACDE_Data/WSN/Grid/Sol') == False:
        os.makedirs('./ACDE_Data/WSN/Grid/Sol')
    Dims = [32, 64, 84]
    for Dim in Dims:
        main_WSN(Dim)




