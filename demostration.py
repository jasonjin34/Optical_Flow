import matplotlib.pyplot as plt
import numpy as np

'''
X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
U, V = np.meshgrid(X, Y)

fig, ax = plt.subplots()
ax.set_title('Moving Forward')
q = ax.quiver(X, Y, U, V)
plt.show()
'''

fig, ax = plt.subplots()

x_pos = [0, 0, 1]
y_pos = [0, 0, 1]
x_direct = [1, 0, 0]
y_direct = [1, -1, 0]


ax.quiver(x_pos,y_pos,x_direct,y_direct,
         scale=5)
ax.axis([-1.5, 1.5, -1.5, 1.5])



plt.show()