module NetworkLoader

using Serialization, SHA

function hash_network(network)::String
    io = IOBuffer()
    serialize(io, network)
    return bytes2hex(sha1(take!(io)))
end

function save_network(network, filename::String)
    open(filename, "w") do io
        serialize(io, network)
    end
end

# Deserialize and load the network from a file
function load_network(filename::String)
    open(filename, "r") do io
        network = deserialize(io)
        return network
    end
end

end