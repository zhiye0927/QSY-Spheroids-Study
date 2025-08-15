import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from matplotlib.colors import LinearSegmentedColormap


colors = ["#66c2a5", "white", "orange"]
cmap = LinearSegmentedColormap.from_list("GreenOrangeCustom", colors, N=256)

l, m = 2, 2

phi = np.linspace(0, 2 * np.pi, 200)
theta = np.linspace(0, np.pi, 100)
phi, theta = np.meshgrid(phi, theta)

Y_lm = sph_harm(m, l, phi, theta)
r = np.real(Y_lm)
r_abs = np.abs(r)

x = r_abs * np.sin(theta) * np.cos(phi)
y = r_abs * np.sin(theta) * np.sin(phi)
z = r_abs * np.cos(theta)

norm = plt.Normalize(-r.max(), r.max())
colors_mapped = cmap(norm(r))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, facecolors=colors_mapped, rstride=1, cstride=1, linewidth=0, antialiased=False)

ax.set_axis_off()
ax.set_box_aspect([1, 1, 1])

max_range = r_abs.max()
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

plt.title(f"Spherical Harmonic $Y_{{{l}}}^{{{m}}}$ Real Part with Custom Colors", fontsize=16, pad=20)
plt.show()