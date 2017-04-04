import numpy as np
import matplotlib.pyplot as plt

files = [
    "selfish-True-0-50-1486863268.pickle",
    "selfish-True-10-50-1486862377.pickle",
    "selfish-True-20-50-1486861420.pickle",
    "selfish-True-30-50-1486860339.pickle",
    "selfish-True-40-50-1486859283.pickle",
    "selfish-True-50-50-1486857877.pickle",
    "selfish-True-60-50-1486856237.pickle",
    "selfish-True-70-50-1486855091.pickle",
    "selfish-True-80-50-1486853954.pickle",
    "selfish-True-90-50-1486852630.pickle",
    "selfish-True-100-50-1486851494.pickle",
    "selfish-True-110-50-1486864319.pickle",
    "selfish-True-120-50-1486866438.pickle",
    "selfish-True-130-50-1486867516.pickle",
    "selfish-True-140-50-1486870032.pickle",
    "selfish-True-150-50-1486871222.pickle",
    "selfish-True-160-50-1486872396.pickle",
    "selfish-True-170-50-1486873400.pickle",
    "selfish-True-180-50-1486874587.pickle",
    "selfish-True-190-50-1486875912.pickle",
    "selfish-True-200-50-1486877142.pickle",
]

results = [np.load(f) for f in files]

def time_to_goal(xs):
    for i, x in enumerate(xs):
        if x[0] < -.13:
            return i

def dist(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def min_dist(r):
    return min([dist(x, y) for (x, y) in zip(r[1][0], r[1][1])])

def total_reward(r):
    def left_lane(t, x, u):
        return -50.*(x[0]+0.13)**2

    return sum([left_lane(None, x, None) for x in r[1][1]])

times_to_goal = [time_to_goal(r[1][1]) for r in results]

print("Time until goal completion ", times_to_goal)

min_distance = [min_dist(r) for r in results]

print("Minumum distance between cars", min_distance)

rewards = [total_reward(r) for r in results]
print("Total Rewards ", )
plt.stem(rewards)
plt.show()

