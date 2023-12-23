module CNN

using ..ActivationFunctions

export FullyConnected, Convolution

abstract type Layer end

mutable struct FullyConnected <: Layer
    input_num::UInt16
    output_num::UInt16
    weights::Array{Float64, 2}
    biases::Vector{Float64}
    activation_function::Function
    function FullyConnected(input_num::UInt16, output_num::UInt16, activation_function::Function)
        if activation_function === ActivationFunctions.relu
            weights = randn(input_num, output_num) .* sqrt(2 / input_num)
        elseif activation_function === ActivationFunctions.leaky_relu
            weights = randn(input_num, ouput_num) .* sqrt(2 / ((1 + 0.01^2) * (input_num + output_num)))
        else
            weights = randn(input_num, output_num) .* sqrt(1 / input_num)
        end
        biases = zeros(output_num)
        new(input_num, output_num, weights,biases)
    end
end

# Convolution Layer, for now, without a bias
mutable struct Convolution <: Layer
    input_channels::UInt16
    output_channels::UInt16
    stride::UInt8
    kernel_size::UInt8
    padding::Bool
    padding_type::String
    kernels::Array{Float64,4}
    activation_function::Function

    function Convolution(input_channels::UInt16, output_channels::UInt16, stride::UInt8, kernel_size::UInt8, padding::Bool, padding_type::String, activation_function::Function)
        if activation_function === ActivationFunctions.relu
            kernels = randn(kernel_size, kernel_size, input_channels, output_channels) .* sqrt(2.0 / (input_channels * kernel_size^2))
        elseif activation_function === ActivationFunctions.leaky_relu
            kernels = randn(kernel_size, kernel_size, input_channels, output_channels) .* sqrt(2.0 / ((1 + 0.01^2) * input_channels * kernel_size^2))
        else
            kernels = randn(kernel_size, kernel_size, input_channels, output_channels) .* sqrt(1.0 / (input_channels * kernel_size^2))
        end
        new(input_channels, output_channels, stride, kernel_size, padding, padding_type, kernels, activation_function)
    end
end

end