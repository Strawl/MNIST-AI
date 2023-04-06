module MNISTData
using HTTP, Downloads, ImageInTerminal, Random


export test_images, train_images, test_labels, train_labels, get_test_batch, get_train_batch

data_dir = "./data"
urls_filenames = [
    ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train_labels"),
    ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train_images"),
    ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "test_images"),
    ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "test_labels")
]

if !isdir(data_dir)
    mkdir(data_dir)
end

for (url, filename) in urls_filenames
    filepath = joinpath(data_dir, filename)
    if !isfile(filepath)
        println("Downloading $filename from $url")
        Downloads.download(url, filepath)
    else
        println("$filename already exists. Skipping download.")
    end
end

test_images = permutedims(reshape(read("./data/test_images")[17:10000*784+16],(784,10000)))
train_images = permutedims(reshape(read("./data/train_images")[17:60000*784+16],(784,60000)))
test_labels = permutedims(read("./data/test_labels")[9:10000+8])
train_labels = permutedims(read("./data/train_labels")[9:60000+8])


function get_train_batch(batch_size)
    indices = randperm(size(train_images, 1))[1:batch_size]
    return collect(zip(Int.(train_labels[indices]), eachrow(normalize_data(train_images[indices,:]))))
end


function get_test_batch(start, amount, test::Bool=false)
    return collect(zip(collect(start:start+amount-1), Int.(test_labels[start:start+amount-1]), eachrow(normalize_data(test_images[start:start+amount-1,:]))))
end



function display_image(num::Int64,test::Bool=false)
    if !test
        img = colorview(Gray,normedview(permutedims(reshape(MNISTData.train_images[num,:],(28,28)))))
    else
        img = colorview(Gray,normedview(permutedims(reshape(MNISTData.test_images[num,:],(28,28)))))
    end
    display(img)
end


function normalize_data(data)
    mean_data = mean(data)
    std_data = std(data)
    return (data .- mean_data) ./ std_data
end

end