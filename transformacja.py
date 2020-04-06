import datetime
import numpy as np
import matplotlib.pyplot as plt


def sigma(x_f):
    sig = 1 / (1 + np.exp(-x_f))
    return sig


start_time = datetime.datetime.now()

# Ilosc wejsc")
N = 4

# Ilosc wyjsc")
K = 4

# Wektory wejść
x = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Ilość danych do szkolenia")
M = 4

# Wektory wyjsc
d = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

H = int(input("Podaj ilosc neuronow ukrytych"))
# H = 1

z = []

# Losowanie wag
w1 = 2 * np.random.rand(H, N) - 1
w2 = 2 * np.random.rand(K, H) - 1

is_bias = input("Czy chcesz przeprowadzić naukę z biasem [True/False]: ")
is_bias = bool(is_bias)

if is_bias:
    bias = 2 * np.random.rand(H) - 1
    str_bias = "\u2714"
else:
    bias = 0
    str_bias = "\u2718"

E = np.zeros((M, 1))

eta = float(input("Podaj współczynnik nauki, najlepiej z przedziału (polecam 0.1): "))
# eta = 0.1
coeff_momentum = float(input("Podaj współczynnik momentum, przedział (0.0 ; 0.9): "))
# coeff_momentum = 0.0

momentum_w2 = np.zeros((K, H))
momentum_w1 = np.zeros((H, K))
momentum_bias = np.zeros(H)

errors = []

epochs = 100_000

for epoch in range(epochs):

    ERROR = 0.0
    z = []

    for m in range(M):
        z_h = sigma(np.dot(w1, x[m]) + bias)
        z.append(z_h)
        # z = sigma(np.dot(w1, x[m]))
        y = sigma(np.dot(w2, z[m]))

        # Funkcja kosztu
        E[m] = np.sum((y - d[m]) * (y - d[m]))
        ERROR += E[m]

        # Obliczanie b
        b2 = []
        for i in range(4):
            b2.append(np.dot((y[i] - d[m][i]), (y[i] * (1 - y[i]))))
        b2 = np.asarray(b2)

        # Obliczanie gradientu dla warstwy ostatniej
        dE_dw2 = []
        for i in range(4):
            var_local = []
            for h in range(H):
                var_local.append(np.dot(np.dot((y[i] - d[m][i]), (y[i] * (1 - y[i]))), z[m][h]))
            var_local = np.asarray(var_local)
            dE_dw2.append(var_local)
        dE_dw2 = np.asarray(dE_dw2)

        dE_dw1 = np.zeros((H, 4))
        dE_db1 = np.zeros(H)
        for i in range(H):
            sum = 0.0
            var_local = []
            for j in range(4):
                sum += np.dot(b2[j], w2[j][i])
            for j in range(4):
                dE_dw1[i][j] = np.dot(np.dot(sum, (z[m][i] * (1 - z[m][i]))), x[m][j])
            dE_db1[i] = np.dot(sum, np.dot(z[m][i], (1 - z[m][i])))

        # punkt kontrolny
        momentum_w2 = dE_dw2 * eta + coeff_momentum * momentum_w2
        momentum_w1 = dE_dw1 * eta + coeff_momentum * momentum_w1
        momentum_bias = dE_db1 * eta + coeff_momentum * momentum_bias

        w2 -= momentum_w2
        w1 -= momentum_w1
        if is_bias:
            bias -= momentum_bias

    ERROR /= 2
    errors.append(ERROR)

end_time = datetime.datetime.now()

######################################  Rysowanie  ################################################################

plt.text(x=(epochs * 0.6), y=0.9, s=(r'$\eta=' + str(eta) + '$\n' +
                                     'ukryte neurony=' + str(H) + '\n' +
                                     'czas nauki=' + str((end_time - start_time).seconds) + 's\n' +
                                     'bias:' + str_bias + '\n' +
                                     'momentum=' + str(coeff_momentum)))
plt.ylabel("Błąd")
plt.xlabel("Epoki")
plt.axis(ymin=0, ymax=2, xmin=0, xmax=epochs)
plt.plot(errors)
plt.grid(b=True)
plt.savefig(f"Plots/neurons({H})_bias({is_bias})_momentum({coeff_momentum}).png")
plt.show()

#######################################  Testowanie  ##############################################################

test_array = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
]

for test in test_array:
    print(test, np.around(sigma(np.dot(w2, sigma(np.dot(w1, test) + bias))), decimals=2))
