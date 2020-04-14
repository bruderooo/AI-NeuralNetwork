import datetime
import numpy as np
import matplotlib.pyplot as plt


def sigma(x_f):
    sig = 1.0 / (1.0 + np.exp(-x_f))
    return sig


data = np.loadtxt("approximation_train_1.txt")

# Ilość danych do szkolenia")
M = len(data)

# H = int(input("Podaj ilosc neuronow ukrytych"))
H = 8

# Losowanie wag
w1 = 2 * np.random.rand(H) - 1
w2 = 2 * np.random.rand(H) - 1

# is_bias = input("Czy chcesz przeprowadzić naukę z biasem [True/False]: ")
# is_bias = bool(is_bias)
is_bias = True

if is_bias:
    bias1 = 2 * np.random.rand(H) - 1
    bias2 = 2 * np.random.rand(1) - 1
    str_bias = "\u2714"
else:
    bias1 = 0
    bias2 = 0
    str_bias = "\u2718"

epochs = 1

errors = []

eta = 10e-10

# Rozpoczęcie nauki
start_time = datetime.datetime.now()
for epoch in range(epochs):

    ERROR = 0.0

    # input_and_target is tab with input and output
    # [0] is input
    # [1] is output
    for input_and_target in data:
        z = sigma(w1 * input_and_target[0] + bias1)
        y = np.dot(w2, z) + bias2[0]

        # Funkcja kosztu
        ERROR += np.sum((y - input_and_target[1]) * (y - input_and_target[1])) / 2

        # Obliczanie gradientu dla warstwy ostatniej
        dE_dw2 = (y - input_and_target[1]) * y * z
        dE_db2 = (y - input_and_target[1]) * y

        # Dla warstwy przedostatniej
        sum = (y - input_and_target[1]) * y * w2
        dE_dw1 = sum * z * (1 - z) * input_and_target[0]
        dE_db1 = sum * z * (1 - z)

        w2 -= dE_dw2 * eta
        w1 -= dE_dw1 * eta
        if is_bias:
            bias1 -= dE_db1
            bias2 -= dE_db2

    ERROR /= M
    errors.append(ERROR)

end_time = datetime.datetime.now()

######################################  Rysowanie  ################################################################

plt.text(x=(epochs * 0.6), y=0.45, s=(r'$\eta=' + str(eta) + '$\n' +
                                      'ukryte neurony=' + str(H) + '\n' +
                                      'czas nauki=' + str((end_time - start_time).seconds) + 's\n' +
                                      'bias:' + str_bias + '\n'
                                      # 'momentum=' + str(coeff_momentum)
                                      ))
plt.ylabel("Błąd")
plt.xlabel("Epoki")
plt.axis(ymin=0, ymax=2, xmin=0, xmax=epochs)
plt.plot(errors)
plt.grid(b=True)
# plt.savefig(f"Plots/neurons({H})_bias({is_bias})_momentum({coeff_momentum}).png")
# plt.show()

#######################################  Testowanie  ##############################################################

test_array = np.loadtxt("approximation_test.txt")

# print(test_array)

for test in test_array:
    print(test[1], np.dot(w2, sigma(w1 * test[0] + bias1)) + bias2[0])
