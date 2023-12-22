module ActivationFunctions

# Activation function for hidden layers
export relu
function relu(x)
    return max(0, x)
end

# Derivative of the ReLU activation function
export relu_prime
function relu_prime(x)
    return x > 0 ? 1 : 0
end

# Leaky ReLU activation function
export leakyrelu
function leakyrelu(x, alpha=0.01)
    return max(alpha * x, x)
end

# Derivative of the Leaky ReLU activation function
export leakyrelu_prime
function leakyrelu_prime(x, alpha=0.01)
    return x > 0 ? 1 : alpha
end

    
end