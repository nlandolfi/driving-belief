import numpy as np

import world
import feature

files = [
    "ndorsa-False-100-30-1486767284.pickle",
    "ndorsa-False-100-50-1486767877.pickle",
    "ndorsa-False-100-100-1486524531.pickle",
    "ndorsa-True-100-30-1486529776.pickle",
    "ndorsa-True-100-50-1486530929.pickle",
    "ndorsa-True-100-100-1486528165.pickle",
]

files = ["./selfish/" + f for f in files]

results = [np.load(f) for f in files]

def entropy(b):
    return sum([i*np.log(i) for i in b])

def info_gain(r, l):
    #print(l)
    info = 0
    for i, b in enumerate(r[2][1]):
        if i == 0:
            continue

        info += (entropy(r[2][1][i-1]) - entropy(b))
    return info

def time_to_goal(xs):
    for i, x in enumerate(xs):
        if x[0] < -.13:
            return i

def dist(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def min_dist(r):
    return min([dist(x, y) for (x, y) in zip(r[1][0], r[1][1])])

def goal_reward(r):
    def left_lane(t, x, u):
        return -50.*(x[0]+0.13)**2
    return sum(left_lane(None, x, None) for x in r[1][1])

# THIS THE WORLD DEFINITION
"""
def dorsa(active=True, theta_explore=100., theta_exploit=1., human_y=0.0):
    theta_normal     = [1., -50., 10., 100., 10., -50.]
    theta_no_speed   = [1., -50., 10., 100., 0. , -50.]
    theta_aggressive = [1., -50., 10., 100., 30., -50.]
    theta_timid      = [1., -50., 10., 100., 5. , -50.]
    theta_distracted = [1., -50., 10., 100., 10., -20.]
    theta_attentive  = [1., -50., 10., 100., 10., -70.]
    theta_distracted1= [1., -50., 10., 100., 1., 0.]
    theta_distracted2= [1., -50., 10., 100., 10., -10.]

    T = 5
    dyn = dynamics.CarDynamics(0.1)
    world = highway()
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.13, human_y, math.pi/2., 1.], color='red', T=T))
    world.cars.append(car.BeliefOptimizerCar(dyn, [0.0, 0.2, math.pi/2., .8], color='yellow', T=T))
    world.cars[1].human = world.cars[0]
    world.cars[0].reward = world.features(theta_attentive, world.cars[0], 'linear')
    world.cars[1].add_model(lambda traj: world.features(theta_attentive, world.cars[0], traj))
    world.cars[1].add_model(lambda traj: world.features(theta_distracted, world.cars[0], traj), log_p=3.)
    @feature.feature
    def left_lane(t, x, u):
        return -(x[0]+0.13)**2
    obj0 = world.cars[1].traj.total(world.features(theta_normal, world.cars[1], 'linear')+theta_exploit*left_lane)
    if active:
        world.cars[1].objective = lambda traj_h: theta_explore*world.cars[1].entropy(traj_h)+obj0
    else:
        world.cars[1].objective = obj0
    return world
"""

def total_reward(r, explore, exploit):
    w = world.dorsa()
    theta_normal     = [1., -50., 10., 100., 10., -50.]
    @feature.feature
    def left_lane(t, x, u):
        return -(x[0]+0.13)**2
    obj0 = w.features(theta_normal, w.cars[1], 'linear')+exploit*left_lane

    us = r[0]
    xs = r[1]
    xs[0].insert(0, w.cars[0].x)
    xs[1].insert(0, w.cars[1].x)

    total_reward = 0.

    for t in range(40):
        #print("Accumulating: ", total_reward)
        for i, car in enumerate(w.cars):
            car.data0['x0'] = xs[i][t]
            car.reset()
            car.u = us[i][t]

        for i in range(len(w.cars)):
            assert np.allclose(xs[i][t], w.cars[i].x)

        total_reward += obj0(t%5, xs[1][t], us[1][t]).eval()

    return total_reward

times_to_goal = [time_to_goal(r[1][1]) for r in results]

print("Time until goal completion ", times_to_goal)

min_distance = [min_dist(r) for r in results]

print("Minumum distance between cars", min_distance)

#print(total_reward(results[5], 100, 30))

table_files = [
    "./dum/selfish-True-0-50-1486863268.pickle",
    "./dum/selfish-distracted-True-0-50-1486967568.pickle",
    "./dum/selfish-True-16-50-1487045781.pickle",
    "./dum/selfish-distracted-True-16-50-1487136587.pickle"
]

table_results = [np.load(f) for f in table_files]

print([goal_reward(r) for r in table_results])
print([total_reward(r, 0, 50.) for r in table_results])

1/0
new_selfish_fs = [
"selfish-True-0-50-1486863268.pickle",
"selfish-True-2-50-1487046843.pickle",
"selfish-True-4-50-1487049015.pickle",
"selfish-True-6-50-1487044711.pickle",
"selfish-True-8-50-1487051731.pickle",
"selfish-True-10-50-1486862377.pickle",
"selfish-True-12-50-1487047850.pickle",
"selfish-True-14-50-1487050328.pickle",
"selfish-True-16-50-1487045781.pickle",
"selfish-True-18-50-1487052705.pickle",
"selfish-True-20-50-1486861420.pickle",
        ]

new_selfish_fs = ["dum/" + f for f in new_selfish_fs]

new_selfish = [np.load(f) for f in new_selfish_fs]

# phew all 40
#print([len(r[0][0]) for r in new_selfish])

new_selfish_rewards = []

for lamb, r in zip([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], new_selfish):
    #w = goal_reward(r)
    w = total_reward(r, 0, 50.)
    #w = r[2][1][39][0]
    #w = info_gain(r, lamb)
    print(w)
    new_selfish_rewards.append(w)

np.save("info_gain", new_selfish_rewards)

1/0

other_files = [
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

other_files = ["fish/"+f for f in other_files]
other_results = [np.load(f) for f in other_files]

fish_rewards = []

for r in other_results:
    w = goal_reward(r)#, 0, 50.)
    print(w)
    fish_rewards.append(w)
np.save("fish_goal_rewards", fish_rewards)
