import numpy as np

# ==========================================
# Activation Functions
# ==========================================

def tanh(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z)**2

def identity(z):
    return z

def identity_deriv(z):
    return np.ones_like(z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def swish(z):
    return z * sigmoid(z)

def swish_deriv(z):
    s = sigmoid(z)
    f = z * s
    return s + f * (1 - s)
