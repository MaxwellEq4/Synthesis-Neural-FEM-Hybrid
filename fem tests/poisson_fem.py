from fem_toolbox import FEM
import numpy as np
import plotly.figure_factory as ff

import plotly.graph_objects as go
from plotly.subplots import make_subplots
#%%



# Initialize FEM model
model = FEM(x0=-1, y0=-1, L1=2, L2=2, noelms1=100, noelms2=100)

def qt_function(VX, VY):
    return 2*np.pi**2*np.cos(np.pi*VX)*np.cos(np.pi*VY)

def f_function(VX, VY):
    return np.cos(np.pi * VX) * np.cos(np.pi * VY)
# Solve and plot
result = model.fem_solve(qt_function=qt_function, f_function=f_function, bc_type='neubc')


# 3D visualization of solution

fig = ff.create_trisurf(x=model.VX, y=model.VY, z=result.flatten(),
                         simplices=model.EToV,
                         colormap="Viridis",
                         title="Poisson Problem: 3D visualization of FEM solution",
                         aspectratio=dict(x=1, y=1, z=1),
                         width = 560,height=500)


fig.show()
fig.write_image("poisson_3d_neubc3.png")
fig.write_image("poisson_3d_neubc3.pdf")
fig.write_image("frontpage.pdf")

#%%

# Create the heatmap figure
fig = go.Figure()

# Define the heatmap
heatmap_fig = go.Heatmap(x=model.VX, y=model.VY, z=result.flatten(),
                         colorscale="Viridis", showscale=True)

# Add the heatmap to the figure
fig.add_trace(heatmap_fig)

# Update layout
fig.update_layout(title='Poisson Problem: 2D visualization of solution', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

# Add appropriate axes labels
fig.update_xaxes(title_text="X-axis")
fig.update_yaxes(title_text="Y-axis")

# Show figure
fig.show()

# Save figure
fig.write_image("poisson_2d_neubc.png")
fig.write_image("poisson_2d_neubc.pdf")


#%%

def qt_function(VX, VY):
    return 2*np.pi**2*np.sin(np.pi*VX)*np.sin(np.pi*VY)

def f_function(VX, VY):
    return np.cos(np.pi * VX) * np.cos(np.pi * VY)

# Initialize FEM model
model = FEM(x0=-1, y0=-1, L1=2, L2=2, noelms1=100, noelms2=100)

# Solve and plot
result = model.fem_solve(qt_function=qt_function, f_function=f_function, bc_type='dirbc')





# 3D visualization of solution

fig = ff.create_trisurf(x=model.VX, y=model.VY, z=result.flatten(),
                         simplices=model.EToV,
                         colormap="Viridis",
                         title="Poisson Problem: 3D visualization of solution",
                         aspectratio=dict(x=1, y=1, z=1),
                         width = 500,height=500)


fig.show()
fig.write_image("poisson_3d_dirbc2.png")
fig.write_image("poisson_3d_dirbc2.pdf")


# Create the heatmap figure
fig = go.Figure()

# Define the heatmap
heatmap_fig = go.Heatmap(x=model.VX, y=model.VY, z=result.flatten(),
                         colorscale="Viridis", showscale=True)

# Add the heatmap to the figure
fig.add_trace(heatmap_fig)

# Update layout
fig.update_layout(title='Poisson Problem: 2D visualization of solution', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

# Add appropriate axes labels
fig.update_xaxes(title_text="X-axis")
fig.update_yaxes(title_text="Y-axis")

# Show figure
fig.show()

# Save figure
fig.write_image("poisson_2d_dirbc2.png")
fig.write_image("poisson_2d_dirbc2.pdf")