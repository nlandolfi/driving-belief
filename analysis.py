import numpy as np
import matplotlib.pyplot as plt

explores = [i * 10 for i in range(20)]

files = [
    "g/robotgoal-0-1486421216.pickle",
    "g/robotgoal-10-1486423283.pickle",
    "g/robotgoal-20-1486425360.pickle",
    "g/robotgoal-30-1486427524.pickle",
    "g/robotgoal-40-1486429482.pickle",
    "g/robotgoal-50-1486431421.pickle",
    "g/robotgoal-60-1486436678.pickle",
    "g/robotgoal-70-1486438872.pickle",
    "g/robotgoal-80-1486441939.pickle",
    "g/robotgoal-90-1486444040.pickle",
    "g/robotgoal-100-1486448037.pickle",
    "g/robotgoal-110-1486450656.pickle",
    "g/robotgoal-120-1486452861.pickle",
    "g/robotgoal-130-1486456373.pickle",
    "g/robotgoal-140-1486458501.pickle",
    "g/robotgoal-150-1486460595.pickle",
    "g/robotgoal-160-1486462750.pickle",
    "g/robotgoal-170-1486465101.pickle",
    "g/robotgoal-180-1486467974.pickle",
    "g/robotgoal-190-1486486201.pickle",
]

results = [np.load(file) for file in files]

def max_deviation(xs):
    max_dev = 0;
    for x in xs:
        if x[0] < max_dev:
            max_dev = x[0]
    return max_dev

def timestep_certain(bs):
    for i, b in enumerate(bs):
        if b[0] > .95:
            return i

    return -1

deviations = [max_deviation(result[1][1]) for result in results]
certainty = [timestep_certain(result[2][1]) for result in results]


fig = plt.figure()
plt.stem(explores, deviations, color="blue")
fig.savefig("expvsdev")
fig = plt.figure()
plt.stem(explores, certainty, color="red")
fig.savefig("expvscer")

def integrated_error(bs):
    return sum([1-b[0] for b in bs])

errors = [integrated_error(result[2][1]) for result in results]
fig = plt.figure()
plt.stem(explores, errors)
fig.savefig("expvserr")
