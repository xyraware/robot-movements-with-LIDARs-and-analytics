import numpy as np
import matplotlib.pyplot as plt
from main import Environment

env = Environment()
heightmap = env.heightmap
measured_moisture = env.moisture


def model_moisture(x, y, h, h_min, h_max):
    V0 = 0.3
    k1 = 0.6
    k2 = 0.1
    a, b = 0.4, 0.3
    norm_h = (h - h_min) / (h_max - h_min)
    return V0 + k1 * (1 - norm_h) + k2 * np.sin(a * x) * np.cos(b * y)


h_min, h_max = heightmap.min(), heightmap.max()

cells = env.cells
xs = np.linspace(0, env.size, cells)
ys = np.linspace(0, env.size, cells)
xv, yv = np.meshgrid(xs, ys)

model_m = model_moisture(xv, yv, heightmap, h_min, h_max)

diff = measured_moisture - model_m

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title("Карта высот")
plt.imshow(heightmap, cmap='terrain', origin='lower')
plt.colorbar(label="Высота (м)")

plt.subplot(2, 2, 2)
plt.title("Измеренная влажность")
plt.imshow(measured_moisture, cmap='Blues', origin='lower', vmin=0, vmax=1)
plt.colorbar(label="Влажность")

plt.subplot(2, 2, 3)
plt.title("Модельная влажность")
plt.imshow(model_m, cmap='Blues', origin='lower', vmin=0, vmax=1)
plt.colorbar(label="Влажность (модель)")

plt.subplot(2, 2, 4)
plt.title("Разница (Измеренная − Модельная)")
plt.imshow(diff, cmap='coolwarm', origin='lower')
plt.colorbar(label="Δ влажность")

plt.tight_layout()
plt.show()
