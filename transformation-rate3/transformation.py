import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys


def sigma(x_f):
    sig = 1 / (1 + np.exp(-x_f))
    return sig


if len(sys.argv) == 1:
    H = int(input("Podaj ilosc neuronow ukrytych: "))
    epochs = int(input("Podaj ilość epok: "))
    eta = float(input("Podaj krok: "))
    is_bias = bool(input("Czy uczyć z biasem [True/False]: "))
    coeff_momentum = float(input("Podaj momentum: "))
else:
    H = int(sys.argv[1])
    epochs = int(sys.argv[2])
    eta = float(sys.argv[3])
    if len(sys.argv) >= 5:
        is_bias = bool(sys.argv[4])
        if len(sys.argv) == 6:
            coeff_momentum = float(sys.argv[5])
        else:
            coeff_momentum = 0.0
    else:
        is_bias = False
        coeff_momentum = 0.0

start_time = datetime.datetime.now()

# Ilosc wejsc")
N = 4

# Ilosc wyjsc")
K = 4

# Wektory wejść i wyjść
x = np.loadtxt("transformation.txt")

# Ilość danych do szkolenia")
M = len(x)

# Losowanie wag
w1 = 2 * np.random.rand(H, N) - 1
w2 = 2 * np.random.rand(K, H) - 1

if is_bias:
    bias1 = 2 * np.random.rand(H) - 1
    bias2 = 2 * np.random.rand(K) - 1
    str_bias = "\u2714"
else:
    bias1 = 0
    bias2 = 0
    str_bias = "\u2718"

# Tworzenie list dla momentum
momentum_w1 = np.zeros((H, N))
momentum_w2 = np.zeros((K, H))
momentum_bias1 = np.zeros(H)
momentum_bias2 = np.zeros(K)

errors = []

for epoch in range(epochs):

    ERROR = 0.0

    for m in range(M):
        z = sigma(np.dot(w1, x[m]) + bias1)
        y = sigma(np.dot(w2, z) + bias2)

        # Funkcja kosztu
        ERROR += np.sum((y - x[m]) * (y - x[m])) / 2

        # Obliczanie b
        b2 = []
        for i in range(K):
            b2.append((y[i] - x[m][i]) * (y[i] * (1 - y[i])))
        b2 = np.asarray(b2)

        # Obliczanie gradientu dla warstwy ostatniej
        dE_dw2 = []
        dE_db2 = np.zeros(K)
        for i in range(K):
            var_local = []
            for h in range(H):
                var_local.append((y[i] - x[m][i]) * (y[i] * (1 - y[i])) * z[h])
            var_local = np.asarray(var_local)
            dE_dw2.append(var_local)
            dE_db2[i] = (y[i] - x[m][i]) * (y[i] * (1 - y[i]))
        dE_dw2 = np.asarray(dE_dw2)

        # Dla warstwy przedostatniej
        dE_dw1 = np.zeros((H, K))
        dE_db1 = np.zeros(H)
        for i in range(H):
            sum = 0.0
            var_local = []
            for j in range(K):
                sum += np.dot(b2[j], w2[j][i])
            for j in range(K):
                dE_dw1[i][j] = sum * z[i] * (1 - z[i]) * x[m][j]
            dE_db1[i] = sum * z[i] * (1 - z[i])

        # Momentum update
        momentum_w2 = dE_dw2 * eta + coeff_momentum * momentum_w2
        momentum_w1 = dE_dw1 * eta + coeff_momentum * momentum_w1
        momentum_bias1 = dE_db1 * eta + coeff_momentum * momentum_bias1
        momentum_bias2 = dE_db2 * eta + coeff_momentum * momentum_bias2

        w2 -= momentum_w2
        w1 -= momentum_w1
        if is_bias:
            bias1 -= momentum_bias1
            bias2 -= momentum_bias2

    ERROR /= K
    errors.append(ERROR)

end_time = datetime.datetime.now()

######################################  Rysowanie  ################################################################

plt.text(x=(epochs * 0.6), y=0.45, s=(r'$\eta=' + str(eta) + '$\n' +
                                      'ukryte neurony=' + str(H) + '\n' +
                                      'czas nauki=' + str((end_time - start_time).seconds) + 's\n' +
                                      'bias:' + str_bias + '\n' +
                                      'momentum=' + str(coeff_momentum)))
plt.ylabel("Błąd")
plt.xlabel("Epoki")
plt.axis(ymin=0, ymax=0.6, xmin=0, xmax=epochs)
plt.plot(errors)
plt.grid(b=True)
# plt.savefig(f"Plots/neurons({H})_bias({is_bias})_momentum({coeff_momentum}).png")
plt.show()

#######################################  Testowanie  ##############################################################

test_array = [
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0]
]

for test in test_array:
    print(test, np.around(sigma(np.dot(w2, sigma(np.dot(w1, test) + bias1)) + bias2), decimals=2))
