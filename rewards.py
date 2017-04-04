import numpy as np

import world

def left_lane(t, x, u):
    return -(x[0]+0.13)**2

def total_positional_reward(results, r, interest = 1):
    return sum([r(t, x, u) for t, (x, u) in enumerate(zip(results[1][interest], results[0][interest]))])

def total_reward(results, r, w, interest=1):
    t = 0
    reward = 0

    times = len(results[0][interest])

    for t in range(times):
        print(t)
        for j, car in enumerate(w.cars):
            print("car ", j)
            x = results[1][j][t]
            u = results[0][j][t]
            print(x)
            print(u)

            if j == interest:
                reward += r(t, car.x, u).eval()

            car.u = u

            car.move()

            #assert np.allclose(x, car.x)

    return reward

def reward_for(path, interest = 1):
    filename = path.replace(".pickle", "")
    name = filename.split('/')[-1]

    return total_reward(list(np.load(path)), w.features(world.theta_attentive, w.cars[interest], 'linear')+left_lane, w)

w = world.robotgoal()

trials = [
    "g/robotgoal-0-1486421216.pickle",
    "g/robotgoal-10-1486423283.pickle",
    "g/robotgoal-20-1486425360.pickle",
    "g/robotgoal-30-1486427524.pickle",
    "g/robotgoal-40-1486429482.pickle",
    "g/robotgoal-50-1486431421.pickle",
]

print([total_positional_reward(list(np.load(t)), left_lane) for t in trials])

#print(reward_for("g/robotgoal-0-1486421216.pickle"))
#print(total_positional_reward(list(np.load("g/robotgoal-0-1486421216.pickle")), left_lane))
#print(total_reward(list(np.load("robotgoal-10-1486423283.pickle")), left_lane))
#print(total_reward(list(np.load("robotgoal-20-1486425360.pickle")), left_lane))
#print(total_reward(list(np.load("robotgoal-30-1486427524.pickle")), left_lane))
#print(total_reward(list(np.load("robotgoal-40-1486429482.pickle")), left_lane))
