import numpy as np
files = [
"equal-nudging-True-0-0-1487058548.pickle",
"equal-nudging-True-5-0-1487063411.pickle",
"equal-nudging-True-10-0-1487064169.pickle",
"equal-nudging-True-50-0-1487059426.pickle",
"equal-nudging-True-100-0-1487060717.pickle",
"equal-nudging-True-150-0-1487061637.pickle",
"equal-nudging-True-200-0-1487062635.pickle",
]

results = [np.load(f) for f in files]

np.save("equal_beliefs.npy", [r[2][1][30] for r in results])
