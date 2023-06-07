import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio




#%% Example 1
# Load the data
u_data = np.load("Data_Integral/Test&Pred_u_example_NN_1.npz")
s_data = np.load("Data_Integral/Test&Pred_s_example_NN_1.npz")

# Extract the values
y = u_data['y']
u_exact = u_data['u_exact']
s_y_pred = u_data['s_y_pred']
s_exact = s_data['s_exact']
s_pred = s_data['s_pred']
s_exact = s_exact-1
s_error = s_exact - s_pred
# Create the figure with two subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Example DeepONet approximating", "Error Plot"))

# Subplot 1
fig.add_trace(
    go.Scatter(x=y.flatten(), y=u_exact.flatten(), mode='lines', name=r'$\text{input function:} -\sin(y)$', line=dict(width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_y_pred.flatten(), mode='lines', name=r'$\text{DeepONet Output Derivative}$', line=dict(width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_exact.flatten(), mode='lines', name=r'$\cos(y)-1$', line=dict(width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_pred.flatten(), mode='lines', name=r'$\text{DeepONet Output}$', line=dict(width=3, dash = 'dash')),
    row=1, col=1
)

# Subplot 2
fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_error.flatten(), mode='lines', name=r'$\text{Error s}$', line=dict(width=2)),
    row=1, col=2
)


fig.update_xaxes(title_text="y", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=1)

fig.update_xaxes(title_text="y", row=1, col=2)
fig.update_yaxes(title_text="Error", row=1, col=2)

fig.update_layout(showlegend=True)
# Update the figure size
fig.update_layout(
    autosize=False,
    width=1100,  
    height=500,  
    title=dict(
        text=r"$\text{Example with:} \cos (y)$",
        font=dict(
            size=38,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
    ),
    legend=dict(
        x=0, 
        y=-0.2,  
        orientation="h",
        yanchor="top",
        xanchor="left",
        font=dict(
            size=24,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
    ),
)
fig.update_xaxes=dict(
        title_text="x-axis",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 1,
    )

fig.update_yaxes=dict(
        title_text="y-axis",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 1,
    ),
fig.update_xaxes=dict(
        title_text="x-axis",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 2,
    )
fig.update_yaxes=dict(
        title_text="y-axis",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 2,
    )


fig.show()
pio.write_image(fig, 'Figures/integral_NN_1.png')

#%% Example 2
# Load the data
u_data = np.load("Data_Integral/Test&Pred_u_example_NN_2.npz")
s_data = np.load("Data_Integral/Test&Pred_s_example_NN_2.npz")

# Extract the values
y = u_data['y']
u_exact = u_data['u_exact']
s_y_pred = u_data['s_y_pred']
s_exact = s_data['s_exact']
s_pred = s_data['s_pred']
s_exact = s_exact-2
s_error = s_exact - s_pred
# Create the figure with two subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("Example DeepONet approximating", "Error Plot"))

# Subplot 1
fig.add_trace(
    go.Scatter(x=y.flatten(), y=u_exact.flatten(), mode='lines', name=r'$\text{input function:} \frac{3}{(5-4 \cos(y))} - 1$', line=dict(width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_y_pred.flatten(), mode='lines', name=r'$\text{DeepONet Output Derative}$', line=dict(width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_exact.flatten(), mode='lines', name=r'$-2 \frac{12 \sin(y)}{5-4 \cos(y)}$', line=dict(width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_pred.flatten(), mode='lines', name=r'$\text{DeepONet Output}$', line=dict(width=3, dash = 'dash')),
    row=1, col=1
)

# Subplot 2
fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_error.flatten(), mode='lines', name=r'$\text{Error s}$', line=dict(width=2)),
    row=1, col=2
)


fig.update_xaxes(title_text="y", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=1)

fig.update_xaxes(title_text="y", row=1, col=2)
fig.update_yaxes(title_text="Error", row=1, col=2)

fig.update_layout(showlegend=True)
# Update the figure size
fig.update_layout(
    autosize=False,
    width = 1100,  
    height=600,  
    title=dict(
        text=r"$\text{Example with:} \frac{3}{(5-4 \cos(y))} - 1$",
        font=dict(
            size=38,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
    ),
    legend=dict(
        x=0, 
        y=-0.1,  
        orientation="h",
        yanchor="top",
        xanchor="left",
        font=dict(
            size=14,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
    ),
)
fig.update_xaxes=dict(
        title_text="x-axis",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 1,
    )

fig.update_yaxes=dict(
        title_text="y-axis",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 1,
    ),
fig.update_xaxes=dict(
        title_text="x-axis",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 2,
    )
fig.update_yaxes=dict(
        title_text="y-axis",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 2,
    )


fig.show()
pio.write_image(fig, 'Figures/integral_NN_2.png')

#%% Example 3
# Load the data
u_data = np.load("Data_Integral/Test&Pred_u_example_NN_3.npz")
s_data = np.load("Data_Integral/Test&Pred_s_example_NN_3.npz")

# Extract the values
y = u_data['y']
u_exact = u_data['u_exact']
s_y_pred = u_data['s_y_pred']
s_exact = s_data['s_exact']+1
s_pred = s_data['s_pred']
s_exact = s_exact
s_error = s_exact - s_pred
# Create the figure with two subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("DeepONet Example", "Error Plot"))

# Subplot 1
fig.add_trace(
    go.Scatter(x=y.flatten(), y=u_exact.flatten(), mode='lines', name=r'$\text{input function:} \cos(7 \pi y)$', line=dict(width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_y_pred.flatten(), mode='lines', name=r'$\text{DeepONet Output Derative}$', line=dict(width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_exact.flatten(), mode='lines', name=r'$\frac{\sin(7\pi y)}{7 \pi} - 1$', line=dict(width=2)),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_pred.flatten(), mode='lines', name=r'$\text{DeepONet Output}$', line=dict(width=3, dash = 'dash')),
    row=1, col=1
)

# Subplot 2
fig.add_trace(
    go.Scatter(x=y.flatten(), y=s_error.flatten(), mode='lines', name=r'$\text{Error s}$', line=dict(width=2)),
    row=1, col=2
)

fig.update_xaxes(title_text="y", row=1, col=1)
fig.update_yaxes(title_text="Value", row=1, col=1)

fig.update_xaxes(title_text="y", row=1, col=2)
fig.update_yaxes(title_text="Error", row=1, col=2)

fig.update_layout(showlegend=True)
# Update the figure size
fig.update_layout(
    autosize=False,
    width = 1100,  
    height=500,  
    title=dict(
        text=r"$\text{Example with:} \frac{\sin(7\pi y)}{7 \pi} - 1$",
        font=dict(
            size=38,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
    ),
    legend=dict(
        x=0, 
        y=-0.2,  
        orientation="h",
        yanchor="top",
        xanchor="left",
        font=dict(
            size=24,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
    ),
)
fig.update_xaxes=dict(
        title_text="y",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 1,
    )

fig.update_yaxes=dict(
        title_text="Error",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 1,
    ),
fig.update_xaxes=dict(
        title_text="y",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 2,
    )
fig.update_yaxes=dict(
        title_text="Error",
        title_font=dict(
            size=30,  # Adjust as needed
            family="'Times New Roman', Times, serif",
        ),
        row = 1,
        col = 2,
    )


fig.show()
pio.write_image(fig, 'Figures/integral_NN_3.png')
