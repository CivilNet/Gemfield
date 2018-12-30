import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

fig, ax = plt.subplots()
#ax = plt.gca()
#ax.set_autoscale_on(False)
polygons = []
color = []

c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]


# polygon
my_seg = [[125.12, 539.69, 140.94, 522.43, 100.67, 496.54, 84.85, 469.21, 73.35, 450.52, 104.99, 342.65, 168.27, 290.88, 179.78, 288, 189.84, 286.56, 191.28, 260.67, 202.79, 240.54, 221.48, 237.66, 248.81, 243.42, 257.44, 256.36, 253.12, 262.11, 253.12, 275.06, 299.15, 233.35, 329.35, 207.46, 355.24, 206.02, 363.87, 206.02, 365.3, 210.34, 373.93, 221.84, 363.87, 226.16, 363.87, 237.66, 350.92, 237.66, 332.22, 234.79, 314.97, 249.17, 271.82, 313.89, 253.12, 326.83, 227.24, 352.72, 214.29, 357.03, 212.85, 372.85, 208.54, 395.87, 228.67, 414.56, 245.93, 421.75, 266.07, 424.63, 276.13, 437.57, 266.07, 450.52, 284.76, 464.9, 286.2, 479.28, 291.96, 489.35, 310.65, 512.36, 284.76, 549.75, 244.49, 522.43, 215.73, 546.88, 199.91, 558.38, 204.22, 565.57, 189.84, 568.45, 184.09, 575.64, 172.58, 578.52, 145.26, 567.01, 117.93, 551.19, 133.75, 532.49]]
for seg in my_seg:
    poly = np.array(seg).reshape((int(len(seg)/2), 2))
    polygons.append(Polygon(poly))
    color.append(c)

#p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
#ax.add_collection(p)
#p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
p = PatchCollection(polygons, cmap=matplotlib.cm.jet, alpha=0.4)
p.set_array(100*np.random.rand(1)  )
ax.add_collection(p)
plt.show()