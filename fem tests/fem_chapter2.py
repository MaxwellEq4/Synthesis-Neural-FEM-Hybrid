from fem_toolbox import FEM
import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
import plotly.graph_objects as go

lam1 = 1
lam2 = 1
#%%
# Exercise 2.1
x0 = 0
y0 = 0
L1 = 1
L2 = 1
noelms1 = 4
noelms2 = 3

fem_instance = FEM(x0, y0, L1, L2, noelms1, noelms2)

# VX and VY are already computed during initialization
VX, VY = fem_instance.VX, fem_instance.VY
print("VX:", VX)
print("VY:", VY)

# EToV is also computed during initialization
EToV = fem_instance.EToV
print("EToV:", EToV)

# Exercise 2.2
n = 9
fem_instance.basfun(n, VX, VY, EToV)
delta, abc = fem_instance.delta, fem_instance.abc
print("delta:", delta)
print("abc:", abc)

k = 2
fem_instance.outernormal(n, k)
n1, n2 = fem_instance.n1, fem_instance.n2
print("n1:", n1)
print("n2:", n2)


# Exercise 2.3 
print('--- Exercise 2.3 ---')


# Test case 1:
x0 = 0; y0 = 0; L1 = 1; L2 = 1; noelms1 = 4; noelms2 = 3; lam1 = 1; lam2 = 1

fem = FEM(x0, y0, L1, L2, noelms1, noelms2)
qt = np.zeros(len(fem.VX))

print('--- Test case 1 ---')
# assuming assembly updates self.A and self.b in your class
fem.assembly(fem.VX, fem.VY, fem.EToV, lam1, lam2, qt)
print(fem.b)


# Test case 2:
x0 = -2.5; y0 = -4.8; L1 = 7.6; L2 = 5.9; noelms1 = 4; noelms2 = 3
fem = FEM(x0, y0, L1, L2, noelms1, noelms2)
qt = -6*fem.VX + 2*fem.VY - 2

print('--- Test case 2 ---')
fem.assembly(fem.VX, fem.VY, fem.EToV, lam1, lam2, qt)

B = fem.A.diagonal()
print(B)
print(fem.b)

#%%
# Exercise 2.4

# test case 1
x0 = 0; y0 = 0; L1 = 1; L2 = 1; noelms1 = 4; noelms2 = 3;lam1 = 1; lam2 = 1;
fem = FEM(x0, y0, L1, L2, noelms1, noelms2)
print('--- Test case 1 ---')

qt = np.zeros(len(fem.VX))
fem.assembly(fem.VX, fem.VY, fem.EToV,lam1,lam2, qt)

bnodes = fem.boundarynodes(fem.EToV)
f = np.ones(len(bnodes))
fem.dirbc(bnodes, f, fem.A, fem.b)

# test case 2
x0 = -2.5; y0 = -4.8; L1 = 7.6; L2 = 5.9; noelms1 = 4; noelms2 = 3
fem = FEM(x0, y0, L1, L2, noelms1, noelms2)
print('--- Test case 2 ---')

qt = -6*fem.VX + 2*fem.VY - 2
fem.assembly(fem.VX, fem.VY, fem.EToV,lam1,lam2, qt)

bnodes = fem.boundarynodes(fem.EToV)
f = fem.VX[bnodes]**3 - fem.VX[bnodes]**2 * fem.VY[bnodes] + fem.VY[bnodes]**2 - 1
fem.dirbc(bnodes, f, fem.A, fem.b)
print(fem.b)

#%%


print('----- Exercise 2.5 -----')
print('----- Test case 1 -----')

# Test case 1
x0 = -2.5; y0 = -4.8; L1 = 7.6; L2 = 5.9; noelms1 = 4; noelms2 = 3
fem = FEM(x0, y0, L1, L2, noelms1, noelms2)

u_analytic = fem.VX**3 - (fem.VX**2)*fem.VY + fem.VY**2
qt = -6*fem.VX + 2*fem.VY - 2

fem.assembly(fem.VX, fem.VY, fem.EToV,lam1,lam2, qt)
bnodes = fem.boundarynodes(fem.EToV)
f = fem.VX**3 - (fem.VX**2)*fem.VY + fem.VY**2

fem.dirbc(bnodes, f, fem.A, fem.b)

u_pred = solve(fem.A, fem.b)
error = u_pred - u_analytic
print(u_pred)
print(error)

# Test case 2
print('----- Test case 2 -----')

dof_vec = []
error_vec = []
for p in range(1, 5):
    noelms1 = 2**p
    noelms2 = 2**p

    fem = FEM(x0, y0, L1, L2, noelms1, noelms2)

    u_analytic = fem.VX**2 * fem.VY**2
    qt = -2*fem.VY**2 - 2*fem.VX**2

    fem.assembly(fem.VX, fem.VY, fem.EToV,lam1,lam2, qt)
    bnodes = fem.boundarynodes(fem.EToV)
    f = fem.VX[bnodes]**2 * fem.VY[bnodes]**2

    fem.dirbc(bnodes, f, fem.A, fem.b)

    u_pred = solve(fem.A, fem.b)
    
    error = max(abs(u_pred-u_analytic))
    dof = len(fem.VX)

    error_vec.append(error)
    dof_vec.append(dof)

plt.figure()
plt.plot(dof_vec, error_vec)
plt.xlabel('Degrees of freedom')
plt.ylabel('Error')
plt.grid(True)
plt.title('Convergence plot')


#%%
print('----- Exercise 2.7 -----')
print('----- Test case 1 -----')

x0 = -2.5
y0 = -4.8
L1 = 7.6
L2 = 5.9
noelms1 = 4
noelms2 = 3
lam1 = 1
lam2 = 1
model = FEM(x0, y0, L1, L2, noelms1, noelms2)

u_analytical = 3 * model.VX + 5 * model.VY - 7
qt = np.zeros(len(model.VX))
f = 3 * model.VX[model.bnodes] + 5 * model.VY[model.bnodes] - 7
q = np.zeros(model.beds.shape[0])

for i in range(model.beds.shape[0]):
    n = model.beds[i, 0]
    k = model.beds[i, 1]
    n1, n2 = model.outernormal(n, k)
    q[i] = -(lam1 * 3 * n1 + lam2 * 5 * n2)

A, b = model.assembly(lam1, lam2, qt)
b = model.neubc(model.beds, q, b)
A, b = model.dirbc(f, A, b)

u = np.linalg.solve(A, b)
E = np.max(np.abs(u_analytical - u))

plt.figure(figsize=(12, 6))
plt.triplot(model.VX, model.VY,model.EToV)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('2D triangular plot', fontsize=17)

print('Solution values u_hat_j')
result = u.reshape((noelms2+1, noelms1+1))
print(result)

print('Error vector for test case 1')
print(E)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(model.VX, model.VY, result.flatten(), triangles=model.EToV, cmap='viridis')
ax.set_xlabel('x', fontsize=15)
ax.set_ylabel('y', fontsize=15)
ax.set_zlabel('z', fontsize=15)
ax.set_title('3D visualization of solution', fontsize=17)


fig = ff.create_trisurf(x=model.VX, y=model.VY, z=result.flatten(),
                         simplices=model.EToV,
                         colormap="Viridis",
                         title="3D visualization of solution",
                         aspectratio=dict(x=1, y=1, z=0.3))

fig.show()
fig.write_image("figure1.png")



print('----- Test case 2 -----')

qt = np.zeros(len(model.VX))
f = np.sin(model.VX[model.bnodes]) + np.sin(model.VY[model.bnodes]) - 7
q = np.zeros(model.beds.shape[0])

for i in range(model.beds.shape[0]):
    n = model.beds[i, 0]
    k = model.beds[i, 1]
    n1, n2 = model.outernormal(n, k)
    q[i] = -(lam1 * 3 * n1 + lam2 * 5 * n2)

A, b = model.assembly(lam1, lam2, qt)
b = model.neubc(model.beds, q, b)
A, b = model.dirbc(f, A, b)

print('Error vector for test case 2')
u = np.linalg.solve(A, b)
E = np.max(np.abs(u_analytical - u))
print(E)

#%%


# c) case noelms1 = noelms2 = 6
x0 = -10
y0 = -10
L1 = 2
L2 = 2
noelms1 = 90
noelms2 = 90
lam1 = 1
lam2 = 1

model = FEM(x0, y0, L1, L2, noelms1, noelms2)

qt = 2 * np.pi ** 2 * np.cos(np.pi * model.VX) * np.cos(np.pi * model.VY)
f = np.cos(np.pi * model.VX[model.bnodes]) * np.cos(np.pi * model.VY[model.bnodes])

A, b = model.assembly(lam1, lam2, qt)
A, b = model.dirbc(f, A, b)

result = np.linalg.solve(A, b)

result_reshaped = np.reshape(result, (noelms2+1, noelms1+1))
result_clipped = result_reshaped[:int(noelms2/2+1), int(noelms1/2+1):]
print('-----c)-----')
print(result_clipped)

# 3D visualization of solution
fig = go.Figure(data=[go.Surface(x=model.VX, y=model.VY, z=result_reshaped)])
fig.update_layout(title='3D visualization of solution', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
fig.write_image("figure3.png")
fig = ff.create_trisurf(x=model.VX, y=model.VY, z=result_reshaped.flatten(),
                         simplices=model.EToV,
                         colormap="Viridis",
                         title="3D visualization of solution",
                         aspectratio=dict(x=1, y=1, z=0.3))

fig.show()
fig.write_image("figure4.png")
