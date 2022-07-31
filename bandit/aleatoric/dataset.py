import numpy as np


def generating_function(x):
    if x > (np.pi) and x < 2 * np.pi:
        noise = np.random.normal(0, 1)
        y = noise
    elif x > -np.pi / 2 and x < np.pi / 2:
        y = 2.5 * np.cos(x)
    else:
        y = 0
    return y


def get_data():
    x1 = np.random.uniform(-3.0, 6.0, size=(32))
    x2 = np.random.uniform(-3.0, 6.0, size=(32))
    x = np.concatenate([x1, x2])
    y = []
    for i in range(len(x)):
        y.append(generating_function(x[i]))
    return x, y


def main():
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    x = np.linspace(-3, 6, 1000)
    y = [generating_function(i) for i in x]
    plt.plot(x, y, color="Purple", linewidth=1)
    plt.savefig("1d_function.png")
    plt.close()


if __name__ == "__main__":
    main()
