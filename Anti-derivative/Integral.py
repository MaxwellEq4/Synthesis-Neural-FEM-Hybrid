#%% Dependencies
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.config import config
from jax.experimental.ode import odeint
from DeepONet_models import DeepONet

from functools import partial
from torch.utils import data
import matplotlib.pyplot as plt


#%% Data Generation
N_train = 10000
m = 100 # number of input sensors
P = 1 # number of output sensors
P_train = 1 
length_scale = 0.2 #lenght_scale for the exponential quadratic kernel
key_train = random.PRNGKey(0)  # use different key for generating training data and test data 
config.update("jax_enable_x64", True) # Enable double precision

#%% RBF 
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)
#%% Generate random functions
N_train = 10000
m = 100 # number of input sensors
P_train = 1   # number of output sensors
length_scale = 0.2 #lenght_scale for the exponential quadratic kernel
key_train = random.PRNGKey(0)  # use different key for generating training data and test data 
config.update("jax_enable_x64", True) # Enable double precision

# Sample GP prior at a fine grid
N = 512
gp_params = (1.0, length_scale)
jitter = 1e-10
X = np.linspace(0, 1, N)[:,None]
K = RBF(X, X, gp_params)
L = np.linalg.cholesky(K + jitter*np.eye(N))
gp_sample = np.dot(L, random.normal(key_train, (N,)))

# Create a callable interpolation function  
u_fn = lambda x, t: np.interp(t, X.flatten(), gp_sample)
# Input sensor locations and measurements
x = np.linspace(0, 1, m)
u = vmap(u_fn, in_axes=(None,0))(0.0, x) #vectorize our code to run it in multiple batches simultaneusly (or to evaluate a function simultaneusly)


# Output sensor locations and measurements
y_train = random.uniform(key_train, (P_train*100,)).sort() 
s_train = odeint(u_fn, 0.0, y_train) # Obtain the ODE solution

fig = plt.figure(figsize=(10, 6))
plt.plot(x,u,'k--',label='$u(x)=ds/dx$')
plt.plot(y_train,s_train,'o-',label='$s(x)=s(0)+\int u(t)dt|_{t=y}$')
plt.title('Solving with Runge-Kutta 45',loc='center', fontsize=16)
plt.xlabel('x', fontsize=12);
plt.ylabel('u', fontsize=12);
plt.legend(fontsize = 12,loc = 'lower right')
figname='RK45.png'
plt.savefig(figname,dpi=300, bbox_inches = "tight")

#%%Generate training data
#training data

# Geneate training data corresponding to one input sample
def generate_one_training_data(key, m=100, P=1):
    # Sample GP prior at a fine grid
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(0, 1, N)[:,None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter*np.eye(N))
    gp_sample = np.dot(L, random.normal(key, (N,)))

    # Create a callable interpolation function  
    u_fn = lambda x, t: np.interp(t, X.flatten(), gp_sample)

    # Input sensor locations and measurements
    x = np.linspace(0, 1, m)
    u = vmap(u_fn, in_axes=(None,0))(0.0, x)

    # Output sensor locations and measurements
    y_train = random.uniform(key, (P,)).sort() 
    s_train = odeint(u_fn, 0.0, np.hstack((0.0, y_train)))[1:] # JAX has a bug and always returns s(0), so add a dummy entry to y and return s[1:]

    # Tile inputs
    u_train = np.tile(u, (P,1))

    return u_train, y_train, s_train

# test data
# Geneate test data corresponding to one input sample
def generate_one_test_data(key, m=100, P=100):
    # Sample GP prior at a fine grid
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(0, 1, N)[:,None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter*np.eye(N))
    gp_sample = np.dot(L, random.normal(key, (N,)))

    # Create a callable interpolation function  
    u_fn = lambda x, t: np.interp(t, X.flatten(), gp_sample)

    # Input sensor locations and measurements
    x = np.linspace(0, 1, m)
    u = vmap(u_fn, in_axes=(None,0))(0.0, x)

    # Output sensor locations and measurements
    y = np.linspace(0, 1, P)
    s = odeint(u_fn, 0.0, y)

    # Tile inputs
    u = np.tile(u, (P,1))

    return u, y, s 

key_train = random.PRNGKey(0)  # use different key for generating training data and test data 
config.update("jax_enable_x64", True) # Enable double precision
keys = random.split(key_train,N_train) # Obtain 10000 random numbers

gen_fn = jit(lambda key: generate_one_training_data(key,m,P_train,))
u_train, y_train, s_train = vmap(gen_fn)(keys)

# Reshape the data
u_train = np.float32(u_train.reshape(N_train * P_train,-1))
y_train = np.float32(y_train.reshape(N_train * P_train,-1))
s_train = np.float32(s_train.reshape(N_train * P_train,-1))

# Testing Data
N_test = 1 # number of input samples 
P_test = m   # number of sensors 
key_test = random.PRNGKey(12345) # A different key 

keys = random.split(key_test, N_test)
gen_fn = jit(lambda key: generate_one_test_data(key, m, P_test))
u, y, s = vmap(gen_fn)(keys)

#Reshape the data
u_test = np.float32(u.reshape(N_test * P_test,-1))
y_test = np.float32(y.reshape(N_test * P_test,-1))
s_test = np.float32(s.reshape(N_test * P_test,-1))

# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u # input sample
        self.y = y # location
        self.s = s # labeled data evulated at y (solution measurements, BC/IC conditions, etc.)
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:]
        y = self.y[idx,:]
        u = self.u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs

#%% Evaluation
# initialize
branch_layers = [100,100,100]
trunk_layers = [1,100,100]

model = DeepONet(branch_layers,trunk_layers)

batch_size = 10000
dataset = DataGenerator(u_train, y_train, s_train, batch_size)

model.train(dataset, nIter = 10000)

params = model.get_params(model.opt_state)

s_pred = model.predict_s(params, u_test, y_test)[:,None]

s_y_pred = model.predict_s_y(params, u_test, y_test)

# Compute relative l2 error
error_s = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test) 
error_u = np.linalg.norm(u_test[::P_test].flatten()[:,None] - s_y_pred) / np.linalg.norm(u_test[::P_test].flatten()[:,None]) 
     
print(error_s, error_u)

#%% visualize
idx = 0
index = np.arange(idx*P_test,(idx + 1) * P_test)

# Compute the relative l2 error for one input sample 
error_u = np.linalg.norm(s_test[index, :] - s_pred[index, :], 2) / np.linalg.norm(s_test[index, :], 2) 
error_s = np.linalg.norm(u_test[::P_test][idx].flatten()[:,None] - s_y_pred[index, :], 2) / np.linalg.norm(u_test[::P_test][idx].flatten()[:,None], 2) 

print("error_u: {:.3e}".format(error_u))
print("error_s: {:.3e}".format(error_s))

# Visualizations
# Predicted solution s(y)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y_test[index, :], s_test[index, :], label='Exact s', lw=2)
plt.plot(y_test[index, :], s_pred[index, :], '--', label='Predicted s', lw=2)
plt.xlabel('y')
plt.ylabel('s(y)')
plt.tight_layout()
plt.legend()

plt.subplot(1,2,2)
plt.plot(y_test[index, :], s_pred[index, :] - s_test[index, :], '--', lw=2, label='error')
plt.tight_layout()
plt.legend()
figname='Test&Pred_s.png'
plt.savefig(figname,dpi=300, bbox_inches = "tight")

fig = plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y_test[index, :], u_test[::P_test][idx], label='Exact u', lw=2)
plt.plot(y_test[index, :], s_y_pred[index,:], '--', label='Predicted u', lw=2)
plt.legend()
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(y_test[index, :], s_y_pred[index,:].flatten() - u_test[::P_test][idx] , '--', label='error', lw=2)
plt.legend()
plt.tight_layout()
figname='Test&Pred_u.png'
plt.savefig(figname,dpi=300, bbox_inches = "tight")

#%% -sin(x)
# Input sensor locations and measurements
x = np.linspace(0, 1, m)
u_fn = lambda t, x: -np.sin(x)
u = u_fn(None,x)

# Output sensor locations and measurements
y = random.uniform(key_train, (m,)).sort()
u2 = np.tile(u, 100)
u2 = np.float32(u2.reshape(N_test * P_test,-1))
y = y.reshape(len(y),1)

# Exact values for -sin(x)
s_exact = np.cos(y) # exact s
u_exact = -np.sin(y) # exact u

# Make predictions
s_pred = model.predict_s(params, u2, y)[:,None]
s_y_pred = model.predict_s_y(params, u2, y)

# Compute the relative l2 error for one input sample 
error_s = np.linalg.norm(s_exact - s_pred, 2) / np.linalg.norm(s_exact, 2) 
error_u = np.linalg.norm(u_exact - s_y_pred, 2) / np.linalg.norm(u_exact, 2) 

print("error_s: {:.3e}".format(error_s))
print("error_u: {:.3e}".format(error_u))

# Visualizations
# Predicted solution s(y)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y, s_exact, label='Exact s', lw=2)
plt.plot(y, s_pred, '--', label='Predicted s', lw=2)
plt.xlabel('y')
plt.ylabel('s(y)')
plt.tight_layout()
plt.legend()

plt.subplot(1,2,2)
plt.plot(y, s_pred - s_exact, '--', lw=2, label='error')
plt.tight_layout()
plt.legend()
figname='Test&Pred_s.png'
plt.savefig(figname,dpi=300, bbox_inches = "tight")

# save data to .npz file for the first figure
np.savez("Data_Integral/Test&Pred_s_example1.npz", y=y, s_exact=s_exact, s_pred=s_pred, error=s_pred - s_exact)


fig = plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y, u_exact, label='Exact u', lw=2)
plt.plot(y, s_y_pred, '--', label='Predicted u', lw=2)
plt.legend()
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(y, s_y_pred.flatten() - u_exact , '--', label='error', lw=2)
plt.legend()
plt.tight_layout()
figname='Test&Pred_u.png'
plt.savefig(figname,dpi=300, bbox_inches = "tight")

# save data to .npz file for the second figure
np.savez("Data_Integral/Test&Pred_u_example1.npz", y=y, u_exact=u_exact, s_y_pred=s_y_pred, error=s_y_pred.flatten() - u_exact)

#%% -12*sin(x) / (5-4cos(x))^2
# Input sensor locations and measurements
x = np.linspace(0, 1, m)
u_fn = lambda t, x: - 12*np.sin(x) / (5-4*np.cos(x))**2
u = u_fn(None,x)

# Output sensor locations and measurements
y = random.uniform(key_train, (m,)).sort()
u2 = np.tile(u, 100)
u2 = np.float32(u2.reshape(N_test * P_test,-1))
y = y.reshape(len(y),1)

# Exact values for -12*sin(x) / (5-4cos(x))^2
s_exact = 3/(5-4*np.cos(y))-1  # exact s
u_exact = - 12*np.sin(y) / (5-4*np.cos(y))**2  # exact u

# Make predictions
s_pred = model.predict_s(params, u2, y)[:,None]
s_y_pred = model.predict_s_y(params, u2, y)

# Compute the relative l2 error for one input sample 
error_s = np.linalg.norm(s_exact - s_pred, 2) / np.linalg.norm(s_exact, 2) 
error_u = np.linalg.norm(u_exact - s_y_pred, 2) / np.linalg.norm(u_exact, 2) 

print("error_s: {:.3e}".format(error_s))
print("error_u: {:.3e}".format(error_u))

# Visualizations
# Predicted solution s(y)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y, s_exact, label='Exact s', lw=2)
plt.plot(y, s_pred, '--', label='Predicted s', lw=2)
plt.xlabel('y')
plt.ylabel('s(y)')
plt.tight_layout()
plt.legend()

plt.subplot(1,2,2)
plt.plot(y, s_pred - s_exact, '--', lw=2, label='error')
plt.tight_layout()
plt.legend()
figname='Test&Pred_s.png'
plt.savefig(figname,dpi=300, bbox_inches = "tight")

# save data to .npz file for the first figure
np.savez("Data_Integral/Test&Pred_s_example2.npz", y=y, s_exact=s_exact, s_pred=s_pred, error=s_pred - s_exact)


fig = plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y, u_exact, label='Exact u', lw=2)
plt.plot(y, s_y_pred, '--', label='Predicted u', lw=2)
plt.legend()
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(y, s_y_pred.flatten() - u_exact , '--', label='error', lw=2)
plt.legend()
plt.tight_layout()
figname='Test&Pred_u.png'
plt.savefig(figname,dpi=300, bbox_inches = "tight")

# save data to .npz file for the second figure
np.savez("Data_Integral/Test&Pred_u_example2.npz", y=y, u_exact=u_exact, s_y_pred=s_y_pred, error=s_y_pred.flatten() - u_exact)

#%% cos(7π * x)
# Input sensor locations and measurements
x = np.linspace(0, 1, m)
u_fn = lambda t, x: np.cos(7*np.pi*x)
u = u_fn(None,x)

# Output sensor locations and measurements
y = random.uniform(key_train, (m,)).sort()
u2 = np.tile(u, 100)
u2 = np.float32(u2.reshape(N_test * P_test,-1))
y = y.reshape(len(y),1)

# Exact values for cos(7π * x)
s_exact = np.sin(7*np.pi*y)/(7*np.pi)-1  # exact s
u_exact = np.cos(7*np.pi*y)  # exact u

# Make predictions
s_pred = model.predict_s(params, u2, y)[:,None]
s_y_pred = model.predict_s_y(params, u2, y)

# Compute the relative l2 error for one input sample 
error_s = np.linalg.norm(s_exact - s_pred, 2) / np.linalg.norm(s_exact, 2) 
error_u = np.linalg.norm(u_exact - s_y_pred, 2) / np.linalg.norm(u_exact, 2) 

print("error_s: {:.3e}".format(error_s))
print("error_u: {:.3e}".format(error_u))

# Visualizations
# Predicted solution s(y)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y, s_exact, label='Exact s', lw=2)
plt.plot(y, s_pred, '--', label='Predicted s', lw=2)
plt.xlabel('y')
plt.ylabel('s(y)')
plt.tight_layout()
plt.legend()

plt.subplot(1,2,2)
plt.plot(y, s_pred - s_exact, '--', lw=2, label='error')
plt.tight_layout()
plt.legend()
figname='Test&Pred_s.png'
plt.savefig(figname,dpi=300, bbox_inches = "tight")

# save data to .npz file for the first figure
np.savez("Data_Integral/Test&Pred_s_example3.npz", y=y, s_exact=s_exact, s_pred=s_pred, error=s_pred - s_exact)


fig = plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(y, u_exact, label='Exact u', lw=2)
plt.plot(y, s_y_pred, '--', label='Predicted u', lw=2)
plt.legend()
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(y, s_y_pred.flatten() - u_exact , '--', label='error', lw=2)
plt.legend()
plt.tight_layout()
figname='Test&Pred_u.png'
plt.savefig(figname,dpi=300, bbox_inches = "tight")

# save data to .npz file for the second figure
np.savez("Data_Integral/Test&Pred_u_example3.npz", y=y, u_exact=u_exact, s_y_pred=s_y_pred, error=s_y_pred.flatten() - u_exact)
