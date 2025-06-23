# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Define the function and its gradient
# def f(x, y):
#     return (x - 2)**2 + (y + 3)**2

# def grad_f(x, y):
#     return np.array([2 * (x - 2), 2 * (y + 3)])

# # Gradient descent parameters
# lr = 0.1
# n_iter = 30
# x, y = 0.0, 0.0

# # Store the path
# path = [(x, y)]
# for _ in range(n_iter):
#     grad = grad_f(x, y)
#     x, y = x - lr * grad[0], y - lr * grad[1]
#     path.append((x, y))
# path = np.array(path)

# # Prepare the grid for contour plot
# X, Y = np.meshgrid(np.linspace(-2, 4, 100), np.linspace(-6, 2, 100))
# Z = f(X, Y)

# # Create the plot
# fig, ax = plt.subplots()
# ax.contour(X, Y, Z, levels=30)
# point, = ax.plot([], [], 'ro-', lw=2)

# def animate(i):
#     point.set_data(path[:i+1, 0], path[:i+1, 1])
#     return point,

# ani = FuncAnimation(fig, animate, frames=len(path), interval=200, blit=True)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Gradient Descent Animation')
# plt.show()

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

class GradientDescentAnimator:
    def __init__(self, master):
        self.master = master
        master.title("Gradient Descent Animation")

        # Paramètres d'entrée
        ttk.Label(master, text="Learning rate:").grid(row=0, column=0)
        self.lr_var = tk.StringVar(value="0.1")
        ttk.Entry(master, textvariable=self.lr_var).grid(row=0, column=1)

        ttk.Label(master, text="Iterations:").grid(row=1, column=0)
        self.iter_var = tk.StringVar(value="30")
        ttk.Entry(master, textvariable=self.iter_var).grid(row=1, column=1)

        ttk.Label(master, text="Start x:").grid(row=2, column=0)
        self.x0_var = tk.StringVar(value="0.0")
        ttk.Entry(master, textvariable=self.x0_var).grid(row=2, column=1)

        ttk.Label(master, text="Start y:").grid(row=3, column=0)
        self.y0_var = tk.StringVar(value="0.0")
        ttk.Entry(master, textvariable=self.y0_var).grid(row=3, column=1)

        # Boutons de contrôle
        self.play_btn = ttk.Button(master, text="Play", command=self.play)
        self.play_btn.grid(row=4, column=0, pady=10)
        self.step_btn = ttk.Button(master, text="Step", command=self.step)
        self.step_btn.grid(row=4, column=1, pady=10)
        self.reset_btn = ttk.Button(master, text="Reset", command=self.reset)
        self.reset_btn.grid(row=4, column=2, pady=10)

        # Figure matplotlib
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=3)

        self.animating = False
        self.reset()

    def f(self, x, y):
        return (x - 2)**2 + (y + 3)**2

    def grad_f(self, x, y):
        return np.array([2 * (x - 2), 2 * (y + 3)])

    def compute_path(self):
        lr = float(self.lr_var.get())
        n_iter = int(self.iter_var.get())
        x, y = float(self.x0_var.get()), float(self.y0_var.get())
        path = [(x, y)]
        for _ in range(n_iter):
            grad = self.grad_f(x, y)
            x, y = x - lr * grad[0], y - lr * grad[1]
            path.append((x, y))
        return np.array(path)

    def draw(self):
        self.ax.clear()
        # Contour plot
        X, Y = np.meshgrid(np.linspace(-2, 4, 100), np.linspace(-6, 2, 100))
        Z = self.f(X, Y)
        self.ax.contour(X, Y, Z, levels=30)
        # Path so far
        if self.step_idx > 0:
            self.ax.plot(self.path[:self.step_idx+1, 0], self.path[:self.step_idx+1, 1], 'ro-', lw=2)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Gradient Descent Animation')
        self.canvas.draw()

    def play(self):
        if not self.animating:
            self.animating = True
            self._play_step()

    def _play_step(self):
        if self.animating and self.step_idx < len(self.path) - 1:
            self.step_idx += 1
            self.draw()
            self.master.after(200, self._play_step)
        else:
            self.animating = False

    def step(self):
        if self.step_idx < len(self.path) - 1:
            self.step_idx += 1
            self.draw()

    def reset(self):
        self.animating = False
        self.path = self.compute_path()
        self.step_idx = 0
        self.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = GradientDescentAnimator(root)
    root.mainloop()