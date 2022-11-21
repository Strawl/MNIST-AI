from data.read import Data
def browse_data(data_object, itype):
    while True:
        x = int(input("please give number: "))
        data_object.display_image(itype, x)


if __name__ == '__main__':
    data = Data("data/")
    browse_data(data_object=data,itype="test")
