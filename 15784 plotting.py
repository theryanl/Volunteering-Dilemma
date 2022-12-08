import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(0)

T = 2
R = 1
P = 0

def classic_two_player_nash(T, R, P):
    return 1 - ((T - R) / (T - P))

def classic_n_player_nash(T, R, P, n):
    return 1 - ((T - R) / (T - P)) ** (1 / (n-1))

def social_distance_two_player_nash(T, R, P, d):
    return 1 - ((T - R) / ((T - P) * (1 + (1 - d))))

def social_distance_n_player_nash(T, R, P, n, d):
    return 1 - ((T - R) / ((T - P) * (1 + (1 - d) * (n - 1)))) ** (1 / (n-1))

def plot_n_player_nash():
    upper = 40
    plt.title(f'Mixed Nash Eq. Volunteer Probabilities for n Players')
    plt.xlabel("Number of players")
    plt.ylabel("Volunteer probability")

    legend = []
    for d in [0, 0.25, 0.5, 0.75]:
        y2 = [social_distance_n_player_nash(T, R, P, n, d) for n in range(2, upper, 1)]
        x2 = [n for n in range(2, upper, 1)]
        plt.plot(x2, y2)
        legend.append(f"Social Distance {d}")

    y = [classic_n_player_nash(T, R, P, n) for n in range(2, upper, 1)]
    x = [n for n in range(2, upper, 1)]
    plt.plot(x, y)
    legend.append("Classic")

    plt.legend(legend)
    # plt.show()
    plt.savefig(f'n_player_nash_p.png')

# plot_n_player_nash()

def at_least_one_volunteer_classic_n_player_nash(T, R, P, n):
    return 1 - (((T - R) / (T - P)) ** (n / (n-1)))

def at_least_one_volunteer_social_distance_n_player_nash(T, R, P, n, d):
    return 1 - ((T - R) / ((T - P) * (1 + (1 - d) * (n - 1)))) ** (n / (n-1))

def plot_n_player_nash_at_least_one():
    upper = 40
    plt.title(f'At Least One Volunteer Probabilities for n Players')
    plt.xlabel("Number of players")
    plt.ylabel("At least one volunteer probability")

    legend = []
    for d in [0, 0.25, 0.5, 0.75]:
        y2 = [at_least_one_volunteer_social_distance_n_player_nash(T, R, P, n, d) for n in range(2, upper, 1)]
        x2 = [n for n in range(2, upper, 1)]
        plt.plot(x2, y2)
        legend.append(f"Social Distance {d}")

    y = [at_least_one_volunteer_classic_n_player_nash(T, R, P, n) for n in range(2, upper, 1)]
    x = [n for n in range(2, upper, 1)]
    plt.plot(x, y)
    legend.append("Classic")
    plt.legend(legend)
    # plt.show()
    plt.savefig(f'n_player_nash_at_least_one.png')

# plot_n_player_nash_at_least_one()

def social_distance_n_player_nash_gaussian(T, R, P, n, d, std):
    # create a truncated normal distribution
    myclip_a = 0
    myclip_b = 1
    my_mean = d
    my_std = std
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    player_distances = stats.truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=n)

    # calculate the probability of each player volunteering
    player_probabilities = [1 - ((T - R) / ((T - P) * (1 + (1 - d) * (n - 1)))) ** (1 / (n-1)) for d in player_distances]

    # calculate the probability of at least one player volunteering
    return 1 - np.prod(1 - np.array(player_probabilities))

def classic_n_player_nash_gaussian(T, R, P, n):
    # create a truncated normal distribution
    myclip_a = 0
    myclip_b = 1
    my_mean = 0.5
    my_std = 0.25
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    player_distances = stats.truncnorm.rvs(a, b, loc=my_mean, scale=my_std, size=n)

    # calculate the probability of each player volunteering
    player_probabilities = [1 - ((T - R) / (T - P)) ** (1 / (n-1)) for d in player_distances]

    # calculate the probability of at least one player volunteering
    return 1 - np.prod(1 - np.array(player_probabilities))

def plot_n_player_nash_gaussian():
    upper = 40
    plt.title(f'At Least One Volunteer Probabilities for n Players (Truncated Gaussian)')
    plt.xlabel("Number of players")
    plt.ylabel("At least one volunteer probability")

    legend = []
    for d in [0.25, 0.75]:
        for std in [0.25, 0.5]:
            y2 = [social_distance_n_player_nash_gaussian(T, R, P, n, d, std) for n in range(2, upper, 1)]
            x2 = [n for n in range(2, upper, 1)]
            plt.plot(x2, y2)
            legend.append(f"Social Distance {d}, std {std}")
    
    y = [classic_n_player_nash_gaussian(T, R, P, n) for n in range(2, upper, 1)]
    x = [n for n in range(2, upper, 1)]
    plt.plot(x, y)
    legend.append("Classic")
    plt.legend(legend)
    # plt.show()
    plt.savefig(f'n_player_nash_gaussian.png')

# plot_n_player_nash_gaussian()
