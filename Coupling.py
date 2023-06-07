#%% Dependencies
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, lax
from jax.config import config
from DeepONet import DeepONet
from scipy.interpolate import griddata
from jax.flatten_util import ravel_pytree
from fem_tool_box import FEM
from functools import partial
from torch.utils import data
import matplotlib.pyplot as plt

#%% Functions for creating data.
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

def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return output_scale * np.exp(-0.5 * r2)

def generate_random_boundary(x, output_scale, lengthscales):
    N = len(x)
    # Covariance matrix
    K = RBF(x, x, (output_scale, lengthscales))
    L = np.linalg.cholesky(K + 1e-10*np.eye(N))  # Small amount of noise for numerical stability
    u_boundary = np.dot(L, np.random.normal(size=(N,)))
    return u_boundary

def generate_one_training_data(key, P, fem, output_scale, lengthscales):
    # Define domain
    x = np.linspace(0, 1, P)
    # Generate random boundary conditions
    u_boundary = generate_random_boundary(x, output_scale, lengthscales)
    # Solve Poisson equation using FEM
    u = fem.fem_solve(lambda x, y: u_boundary, lambda x, y: np.full_like(x, 6), bc_type='dirbc')
    # Grid coordinates
    y = np.vstack([fem.VX, fem.VY]).T
    # Solution at grid points
    s = u.flatten()
    # Tile boundary conditions
    u = np.tile(u_boundary, (P**2, 1))

    return u, y, s

def generate_one_test_data(key, P, fem, output_scale, lengthscales):
    # Similar to training data
    u_test, y_test, s_test = generate_one_training_data(key, P, fem, output_scale, lengthscales)
    
    return u_test, y_test, s_test


def generate_training_data(key, N, P):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    
    # Create a FEM instance
    fem_NN = FEM(0.3, 0.3, 0.4, 0.4, 10, 10)  # Using 100 elements as an example
    
    u_train, y_train, s_train = vmap(generate_one_training_data, (0, None))(keys, P, fem_NN)
    u_train = np.float32(u_train.reshape(N * P, -1))
    y_train = np.float32(y_train.reshape(N * P, -1))
    s_train = np.float32(s_train.reshape(N * P, -1))

    config.update("jax_enable_x64", False)
    return u_train, y_train, s_train


def generate_test_data(key, N, P):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    
    # Create a FEM instance
    fem_NN = FEM(0.3, 0.3, 0.4, 0.4, 10, 10)  # Using 100 elements as an example

    u_test, y_test, s_test = vmap(generate_one_test_data, (0, None))(keys, P, fem_NN)
    u_test = np.float32(u_test.reshape(N * P**2, -1))
    y_test = np.float32(y_test.reshape(N * P**2, -1))
    s_test = np.float32(s_test.reshape(N * P**2, -1))

    config.update("jax_enable_x64", False)
    return u_test, y_test, s_test

# Generate a random key for data generation
key = random.PRNGKey(1234)

# Generate training and test data
N = 1000  # number of training cases
P = 100  # number of sensor locations for each training case

u_train, y_train, s_train = generate_training_data(key, N, P)
u_test, y_test, s_test = generate_test_data(key, N, P)

# Create DataGenerator instances for training and testing datasets
batch_size = 10000
train_dataset = DataGenerator(u_train, y_train, s_train, batch_size)
test_dataset = DataGenerator(u_test, y_test, s_test, batch_size)

# Define the DeepONet architecture
branch_layers = [100, 100, 100]
trunk_layers = [1, 100, 100]

# Initialize DeepONet model
DeepONet_model = DeepONet(branch_layers, trunk_layers)

# Train the model
nIter = 10000  # number of training iterations
DeepONet_model.train(train_dataset, nIter)

# Fetch the parameters of the trained model
params = DeepONet_model.get_params(DeepONet_model.opt_state)

# Predict the solutions with the trained model
s_pred = DeepONet_model.predict_s(params, u_test, y_test)[:, None]
s_y_pred = DeepONet_model.predict_s_y(params, u_test, y_test)

# Compute relative L2 error
error_s = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
error_u = np.linalg.norm(u_test[::P].flatten()[:, None] - s_y_pred) / np.linalg.norm(u_test[::P].flatten()[:, None])

print(error_s, error_u)


