import numpy as np
import matplotlib.pyplot as plt

files = [
    "nudging-True-0-0-1486866715.pickle",
    "nudging-True-10-0-1486865988.pickle",
    "nudging-True-20-0-1486864896.pickle",
    "nudging-True-30-0-1486863780.pickle",
    "nudging-True-40-0-1486861812.pickle",
    "nudging-True-50-0-1486860323.pickle",
    "nudging-True-60-0-1486858949.pickle",
    "nudging-True-70-0-1486857797.pickle",
    "nudging-True-80-0-1486855957.pickle",
    "nudging-True-90-0-1486854918.pickle",
    "nudging-True-100-0-1486852988.pickle",
    "nudging-True-110-0-1486867674.pickle",
    "nudging-True-120-0-1486869420.pickle",
    "nudging-True-130-0-1486870512.pickle",
    "nudging-True-140-0-1486871478.pickle",
    "nudging-True-150-0-1486873152.pickle",
    "nudging-True-160-0-1486875115.pickle",
    "nudging-True-170-0-1486885691.pickle",
    "nudging-True-180-0-1486886832.pickle",
    "nudging-True-190-0-1486888263.pickle",
    "nudging-True-200-0-1486889259.pickle",
]

results = [np.load(f) for f in files]

def max_deflection(xs):
    return -min([x[0] for x in xs])

def avg_deflection(xs):
    return np.mean([x[0] for x in xs])

def max_belief(bs):
    return max([b[0] for b in bs])

#plt.stem([r[2][1][20][0] for r in results])
#plt.show()

deflections = [max_deflection(r[1][1]) for r in results]
#plt.stem(deflections)
#plt.show()
beliefs = [max_belief(r[2][1]) for r in results]
#plt.stem(beliefs)
#plt.show()

"""
"rss_merge-0.000000-1486970967.pickle",
"rss_merge-0.050000-1486971344.pickle",
"rss_merge-0.100000-1486971701.pickle",
"rss_merge-0.150000-1486972030.pickle",
"rss_merge-0.200000-1486972340.pickle",
"rss_merge-0.250000-1486972658.pickle",
"rss_merge-0.300000-1486972986.pickle",
"rss_merge-0.350000-1486973470.pickle",
"rss_merge-0.400000-1486973798.pickle",
"""

r_files = [
"rss_merge--0.200000-1487199552.pickle",
"rss_merge--0.150000-1487199259.pickle",
"rss_merge--0.100000-1487198959.pickle",
"rss_merge--0.050000-1487198652.pickle",
"rss_merge-0.000000-1486970967.pickle",
"rss_merge-0.050000-1486971344.pickle",
"rss_merge-0.100000-1486971701.pickle",
"rss_merge-0.150000-1486972030.pickle",
"rss_merge-0.200000-1486972340.pickle",
"rss_merge-0.250000-1486972658.pickle",
"rss_merge-0.300000-1486972986.pickle",
"rss_merge-0.350000-1486973470.pickle",
"rss_merge-0.400000-1486973798.pickle",
"rss_merge-0.450000-1487199855.pickle",
"rss_merge-0.500000-1487200148.pickle",
"rss_merge-0.550000-1487200443.pickle",
"rss_merge-0.600000-1487200736.pickle",
        ]

rss_results = [np.load(f) for f in r_files]
plt.stem([-max_deflection(r[1][1]) for r in rss_results])
plt.title("Max Deflection")
plt.show()
avg_deflections = [-avg_deflection(r[1][1]) for r in rss_results]
np.save("rss_merge_avg_deflections", avg_deflections)
plt.title("avg Deflection")
plt.show()
