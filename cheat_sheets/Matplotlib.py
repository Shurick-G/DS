import matplotlib.pyplot as plt 
plt.ylim(-50, 500)
plt.xlim(0, 200) 

df.boxplot()
df.boxplot('total_area', 'is_it_piter')
df.boxplot('<Что изучаем>', '<В каком разрезе>')


import matplotlib.pyplot as plt
# После команды вывода графика вызывают метод show(). 
# Он позволяет посмотреть, как отличаются гистограммы с разным числом корзин: 
data.hist(bins=10)
plt.show()
data.hist(bins=100)
plt.show() 

df = pd.DataFrame({'a': [2, 3, 4, 5], 'b': [4, 9, 16, 25]})
print(df)
df.plot() 
df.plot(style='o')  # Только точки
df.plot(style='x')  # Вместо точек – крестики
df.plot(style='o-') # 'o-' - кружок и линия 
# По умолчанию, метод plot строит график используя индексы в качестве значений оси Х
df.plot(x='b', y='a', style='o-') 
df.plot(x='b', y='a', style='o-', xlim=(0, 30)) 
df.plot(x='b', y='a', style='o-', xlim=(0, 30), grid=True) 
df.plot(x='b', y='a', style='o-', xlim=(0, 30), grid=True, figsize=(10, 3)) 



hw.sort_values('height').plot(x='height', y='weight') 
hw.plot(x='height', y='weight', kind='scatter') 
hw.plot(x='height', y='weight', kind='scatter', alpha=0.03) 

station_stat_full.plot.scatter(x='count', y='time_spent',  grid=True)
station_stat_full.plot.scatter(x='count', y='time_spent',  grid=True, alpha=0.1)
# Тоже самое что и:
station_stat_full.plot(x='count', y='time_spent', kind='scatter',  grid=True)


# Когда точек много и каждая в отдельности не интересна, данные отображают особым способом. 
# График делят на ячейки; пересчитывают точки в каждой ячейке. 
# Затем ячейки заливают цветом: чем больше точек — тем цвет гуще.
hw.plot(x='height', y='weight', kind='hexbin', gridsize=20, figsize=(8, 6), sharex=False, grid=True)
# gridsize – число ячеек по горизонтальной оси, аналог bins для hist().
# При столкновении с багами приходится ставить «костыли». 
# Здесь это параметр sharex=False. 
# Если значение True, то пропадёт подпись оси Х,
# а без sharex график выйдет неказистым — это «костыльный» обход бага библиотеки pandas.