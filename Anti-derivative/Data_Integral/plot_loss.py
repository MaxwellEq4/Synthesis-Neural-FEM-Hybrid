import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Load the data
loss_data = np.load('loss_log_integral.npy')

# Create the figure
fig = go.Figure()

# Add line+marker plot
fig.add_trace(go.Scatter(
    x=list(range(len(loss_data))),
    y=loss_data,
    mode='lines+markers',
    name='training loss'
))

# Update layout
fig.update_layout(
    title="Training Loss for Anti-Derivative Problem",
    xaxis_title="Iterations",
    yaxis_title="Loss",
    autosize=False,
    width=600,
    height=400,
    yaxis_type="log",  # make the y-axis logarithmic
)

# Save the figure
pio.write_image(fig, 'training_loss.png')
