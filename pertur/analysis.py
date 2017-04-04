import numpy as np
import matplotlib.pyplot as plt

files = [
    "perturb-y-00-1486775575.pickle",
    "perturb-y-05-1486776658.pickle",
    "perturb-y-10-1486777798.pickle",
    "perturb-y-15-1486779006.pickle",
    "perturb-y-20-1486780219.pickle",
    "perturb-y-25-1486781770.pickle",
    "perturb-y-30-1486783430.pickle",
    "perturb-y-35-1486784830.pickle",
    "perturb-y-40-1486786022.pickle",
]

results = [np.load(f) for f in files]

def max_deflection(xs):
    return -min([x[0] for x in xs])

def dist(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def avg_dist(hx, rx, s):
    return np.mean([(h[1] - r[1]) - s for h, r in zip(hx, rx)])

def time_to_goal(xs):
    for i, x in enumerate(xs):
        if x[0] <= -.12999:
            return i

    return -1

starting = [-.2, -.15, -.1, -0.05, 0, 0.05, 0.1,0.15,0.2]

deflections = [max_deflection(r[1][1]) for r in results]
plt.stem(deflections)
plt.title("Deflection")
plt.show()
np.save("deflections", deflections)

deltas = [avg_dist(r[1][0], r[1][1], s) for r,s in zip(results, starting)]
plt.stem(deltas)
plt.title("Avg Delta y")
plt.show()
np.save("deltas", deltas)

time_to_goals = [time_to_goal(r[1][1]) for r in results]
plt.stem(time_to_goals)
plt.title("time to goals")
np.save("time_to_goals", time_to_goals)
plt.show()

