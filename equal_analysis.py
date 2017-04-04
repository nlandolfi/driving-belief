import numpy as np


equal_files = [
"equal-nudging-True-0-0-1487058548.pickle",
"equal-nudging-True-5-0-1487063411.pickle",
"equal-nudging-True-10-0-1487064169.pickle",
"equal-nudging-True-50-0-1487059426.pickle",
"equal-nudging-True-100-0-1487060717.pickle",
"equal-nudging-True-150-0-1487061637.pickle",
"equal-nudging-True-200-0-1487062635.pickle",
]

equal_files = ["dum/" + f for f in equal_files]

results = [np.load(f) for f in equal_files]
print([r[2][1][20][0] for r in results])
print([r[2][1][30][0] for r in results])
print([r[2][1][39][0] for r in results])
