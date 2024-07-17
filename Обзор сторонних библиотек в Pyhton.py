import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter



print('*********************    class Requests    *********************')

class Requests:

    url = 'https://alfabank.ru/'

    response = requests.get(url)

    if response.status_code == 200:

        data = response.url
        print(f'Статус ответа: OK [код 200]')

    else:
        print('Ошибка при выполнении запроса')



print('*********************    Panda    *********************')

class Panda:

    city = {'Город': ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург'],
        'Год основания': [1147, 1703, 1893, 1723],
        'Население (тыс.чел.)': [11.9, 4.9, 1.5, 1.4]}


    df = pd.DataFrame(city)

    print(df)

print('*********************    Panda    *********************')
print('**********    Загрузка  из текстового файла    ************')

data = pd.read_fwf(r'C:\Users\User\PycharmProjects\Библиотеки\Dostyp.txt')

print(data.head())

print('*********************    Numpy    *********************')

class Numpy:
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    sum = np.sum(arr)
    flip = np.flip(arr)

    print(arr)
    print(arr1)
    print(arr2)
    print(sum)
    print(flip)


print('*********************    Matplotlib    *********************')

class Matplotlib:
    x = [8, 6, 4, 2, 0]
    y = [30, 10, 15, 22, 26]

    plt.plot(x, y)
    plt.xlabel('ось X')
    plt.ylabel('ось Y')
    plt.title('Линейный график')
    plt.show()


print('*********************    Matplotlib    *********************')


class Pillow:
    image = Image.open(r'C:\Users\User\PycharmProjects\Библиотеки\Средства.jpeg')
    resized_image = image.resize((800, 600))
    resized_image.save('resized_image.jpeg')

    image = Image.open(r'C:\Users\User\PycharmProjects\Библиотеки\Средства.jpeg')
    blurred_image = image.filter(ImageFilter.BLUR)
    blurred_image.save('blurred_image.jpeg')

   
    image = Image.open(r'C:\Users\User\PycharmProjects\Библиотеки\Средства.jpeg')
    image.save('converted_image.jpeg')
    image.save('converted_image.gif')