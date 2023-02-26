pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scipy

import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from scipy import stats as st


# Ябучий символ из 1С  

# Операторы Pandas пишут не словами, а знаками: 
# and превращается в & (амперсанд), 
# or превращается в | (вертикальная черта), 
# а not превращается в ~ (тильда).
# Каждое отдельное сравнение необходимо окружать скобками. 
# Например, логическое выражение в Python x > 0 and x < 10 в Pandas превратится в (x > 0) & (x < 10).



# Напоследок о разнице между loc и его сокращенной записью. Вот эти две записи на примере поиска всех работающих над новой функцией:
# полная запись
print(data.loc[data['Новая функция'] == '+'])

# сокращённая запись
print(data[data['Новая функция'] == '+']) 
# Сокращённая запись вам уже встречалась, но в этом уроке мы специально про неё не говорили, чтобы вас не запутать. Её применение ограничено:
# Если мы передаем только условие (как в примере выше), то можно использовать любую из двух записей, они идентичны. Логические операторы тоже можно использовать.
# Если мы хотим указать и условие, и столбец, подойдет только полная запись с loc.
# Если мы хотим изменить значения в таблице, подойдет только полная запись с loc.



# использовать в качестве индекса "наблюдение" (первый столбец).
df = pd.read_csv('<путь к файлу>', index_col=1)  
#Тогда к строкам можно образщаьтся по имени"наблюдения":
df.loc['<Имя наблюдения>']  f.loc['<индекс строки>, <имя столбца>']

LP_registrations.loc[LP_registrations['phone'] != 0, 'Дубль'] = 'Да'

# Можно сначало логически проиндексировать строки и сохранить в переменную (список гоических операторов),
# а потом передать список в .loc[]
rows = (data['Новая функция'] == '+') & (data['Роль'] == 'разработчик')
data.loc[rows, "Роль"] = "улучшатель"


df.iloc[<номер строки>, <номер столбца>  ]
# название столбцов df в список
list(df.columns) 
# Обращение к ячейки датафрэйма
df.at['<индекс строки>', '<название столбца>'] 
df.iat[1, 2] #тоже самое, но по индексам – работае намного бычстрее, т.к. Панда параллелит итерации
#присваевание (в т.ч. и замена) названий столбцов, но только всех сразу
df.columns = ['<название элемента 1>', '<название элемента 2>', '<название элемента 3>', и т.д.]
#Обращения к элементам датафрэйма:
	#По стлбцу
df[0]
	#По срезу строк
df[1:3] # если обращаться по индексам, то третий эллемент не будет включен в срез
df['<первый эллемент>':'<третий эллемент>'] # если обращаться по именам, то третий эллемент попадет в срез
# Если имя столбца – валидный питоновский идентификатор, т.е. могло бы быть имененм переменной, то к столбцам можно обращаться через точку:
df.<имя элемента> # так можно обращаться к столбцу, но не заводить новый столбец
# фильтрация по условию:
df[df['<имя элемента>'] > 3]
# отбор значений по спискку (аналог SQLного "where x in (.......)")
df[df['user_id'].isin(<список или переменная, содержащая список>)]

#среднее по всему датафрэйму:
df.mean()
#среднее по строкам:
df.mean(axis=1)
#среднее по стобцам:
df.mean(axis='columns')
#транспанировать таблицу:
df.T
# Группировка
groups = df.groupby('<имя столбца>') #созжает объект типа groups
# обращение к группе 
groups.get_group('<имя группы>')
#получить название всех групп
groups.groups.keys()
df[<имя столбца>].value_counts() # посчитать уникальное количество значений
#пример периписи значений
df[df['UnitOfMeasure'] == 'руб/1000 куб.м.']['TariffValue'] = df[df['UnitOfMeasure'] == 'руб/1000 куб.м.']['TariffValue'] / 1000



df_reader['email1_fin'] = np.where(df_reader['email1_b'].isnull(), 
                                   df_reader['email1_a'],
                                   df_reader['email1_b'])

df['DataFrame Column'] = df['DataFrame Column'].astype(int)


res_grouped = res_grouped.set_index('Клиент')


#https://habr.com/ru/post/196980/

# Вставить столбец
df2.insert(1,'country', country) 
# В нашем случае функции передается 3 аргумент:
# номер позиции, куда будет вставлен новый столбец
# имя нового столбца
# массив значений столбца (в нашем случае, это обычный список list)

# merege, что-то вроде SQL'ского JOIN'а или MERGE'а из M (PowerQuery)
res = df2.merge(df1, 'left', on='shop')
# В качестве параметров функция принимает:
# набор данных (который будет присоединен к исходному)
# тип соединения
# поле, по которому происходит соединение
result = df_1.merge(df_2, 'left', on='<поле для джоина>') # ещё пример
# ещё пример
counterparty = registrations.merge(
                                     counterparty_by_emails[['Контрагент','СовпадениеПоПоиску', 'ДатаПервогоКонтакта']], 
                                     how='left', 
                                     left_on='email',  
                                     right_on='СовпадениеПоПоиску'
                                     
                                     )


pivot_table()
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html
res.pivot_table(['qty'],['country'], aggfunc='sum', fill_value = 0)
# В нашем примере функция в качестве параметров принимает:
# список столбцов, по которым будет считаться агрегированные значение
# список столбцов, которые будут строками итоговой таблицы
# функция, которая используется для агрегации
# параметр для замены пустых значений на 0


uids = uids.merge(uid_user_id, 'left', on='uid')

uids[(uids['user_id'].notnull())]

new_uids_df['created_date'] = pd.to_datetime(new_uids_df['created_date'])

# вычисление разницы дат в днях (обе даты должны быть в формате datetime)
delta = (row['next_date'] - row['Дата']).days
pd.Timedelta(1, "d")
pd.Timedelta(hours=3)
data['local_time'].round('1H')


# Замена части текста как в Excel через Ctrl + H
feed = feed.replace('https://www.sima-land.ru', '', regex=True)

# отбор значения по условию "Содержит"
to_filter = general[general['source'].str.contains('google', na = False)]

# Замена пустых значений
df = df.fillna('not set')

# Сортировка
df.sort_values(<имя столбца>)
df.sort_index()


df.describe()

df.groupby(['<столбец 1>', '<Столбец 2>']).sum()
df.groupby(['<столбец 1>', '<Столбец 2>']).agg(['sum', 'max'])

logs.groupby('source').agg({'purchase':['count', 'sum']})

# Никогда не используйте циклы в работе с большт количесвом данных
# Метод
.apply(func) 
# позволяет применить функцию ко всем элементам df, не применяя цикла (отрабатывает намного быстрее)
# Метод
.map()
# позволяет применить функцию к Series, работает аналогично apply()
#Пример:
df['Name_lenght'] = df.apply(lambda x: len(x['Name']), axis=1)


# Вариант условия если для построчного прохода
orders.loc[orders['id'] == orders['first_order_id'], 'is_this_new_order'] = 'Yes'
# Удаляем строки без контрагентов (совпвдение только по потенциалу)
counterparty = counterparty[(counterparty['Контрагент_x'] != 'пусто') | (counterparty['Контрагент_y'] != 'пусто')]
# проверка
counterparty[(counterparty['Контрагент_x'] == 'пусто') & (counterparty['Контрагент_y'] == 'пусто')]


LP_registrations.loc[
    LP_registrations['email'] == 0|LP_registrations['phone'] == 0, 
    'is_this_new_order'] = 'Yes' 

LP_registrations[(LP_registrations['email'] == 0) & (LP_registrations['phone'] == 0)]['Дубль'] = 'Да'


df.dtypes()

df['age'].notna() # обрати внимани, что df['age'].notna() – уже не  DataFrame, а Series
df[df['age']].notna() # а вот так буде обработам именно DataFrame


df['age'].isna().sum()

df.loc[df['Age'].notna(), 'Name'] # loc отфильтрует по первому значению, а вернет серию, указанную во втором


 # сортировка
 df.sort_values(['Age', 'Name'], asceding=[False, True])

 # Копирование датафрейма
 df2 = df.copy(deep=True)
 # deep=True – копируется не ссылка а все объекты целиком

 # Объединение датафреймов
 # Первый способ – конкатинация, что-то типа Append ну или Union
    cdf1 = pd.concat([df, df2])
    cdf1 = pd.concat([df, df2], axis=1) # конкотинирование по столбцам
# Способ второй – merge
-----







import seaborn as sns
import matplotlib.pyplot as plot
sns.set(rc={'figure.figsize': (16, 6)}, style='witegrid')





# Фильтры
retail[retail.UnitPrice < 0]
retail[retail.CustomerID.isna()]
retail = retail.fillna({'CustomerID' : 0})
retail[retail.CustomerID == 0]
retail = retail.astype({'CustomerID' : 'Int'})
retail = retail.query('UnitPrice > 0 ')

retail.groupby('country', as_index=False).agg({'CustomerID':'count'}) \
    .sort_values('CustomerID', ascending=False).shape


retail[retail.CustomerID > 0 ].groupby('country',as_index=False) \
    .agg({'CustomerID' : pd.Series.nunique}).sort_values('CustomerID', ascending=False)


UK_data = retail.query("country == 'United Kingdom' & CustomerID > 0")

UK_data.groupby('CustomerID', as_index=False) \
    .agg({'InvoiceNo': 'count'}) \
    .rename(columns={'InvoiceNo': 'transactions'}) \
    .sort_values('transactions', ascending=False).transactions.median()


transactions_data = UK_data.groupby('CustomerID', as_index=False) \
                            .agg({'InvoiceNo' : 'count'}) \
                            .rename(columns={'InvoiceNo': 'transactions'}) \
                            .sort_values('transactions', ascending=False).transactions

sns.distplot(transactions_data.query('transactions < 1000').transactions, kde=False)

retail.invoiceData = pd.to_datetime(retail['invoiceData'])

UK_data.resample('M').CustomerID.nunique()   # M – meens by Month
UK_data.resample('D').CustomerID.nunique()   # D – meens by Day
                                             # W –||– week
UK_data.resample('M').CustomerID.nunique().plot()


UK_data.resample('M').CustomerID.nunique().rolling(10).maen() #  rolling() – скользящее среднее, 10 – окно скольжения
UK_data.resample('M').CustomerID.nunique().rolling(10).maen().plot()




inp_list = ['John', 'Bran', 'Grammy', 'Norah'] 
 
res = ' '.join([str(item) for item in inp_list]) 
print("Converting list to atring using List Comprehension:\n")
print(res) 









# ----------------------------------- Предобработка данных -------------------------------------------
# Посмотреть колдичество пропускоы в df
df.isna().sum() 

# Заполнение пропусков нужным занчением
df['track_name '] = df['track_name'].fillna('unknown') 

# Удаление столбцов, в которых в столбцах total_cases, deaths или case_fatality_rate встречается NaN
cholera = cholera.dropna(subset=['total_cases', 'deaths', 'case_fatality_rate'], axis='columns') 

dfg[~dfg['short_url_cat'].notnull()]


# Поиск явных дубликатов
df.duplicated()
df.duplicated().sum()

# Напомним, что если вызвать метод duplicated() без подсчёта суммы, 
# то на экране будут отображены все строки. Там, где есть дубликаты, будет логическое значение True, 
# где дубликата нет — False. 
# Метод sum() воспринимает все значения True как единицы, поэтому происходит сложение всех единиц, 
# то есть находится количество дубликатов.

duplicated_df = df[df.duplicated()].head() #результат — датафрейм с дубликатами

df = df.drop_duplicates() 
df = df.drop_duplicates().reset_index()          # После удаления строчек лучше обновить индексацию: чтобы в ней не осталось пропусков
df = df.drop_duplicates().reset_index(drop=True) # Можно и не создавать столбец index. Для этого у метода reset_index() изменим специальный параметр
df = df.drop_duplicates(subset=['name'], keep='first')
df = df.drop_duplicates(subset=['name'], keep='last' )

# Способ второй "ручной поиск дубликатов"
value_counts() # анализирует столбец, выбирает каждое уникальное значение и подсчитывает частоту его встречаемости в списке
stock['item'].value_counts()

# Чтобы учесть такие дубликаты, 
# все символы в строках приводят к нижнему регистру вызовом метода lower():
str.lower()
# Жля приведения к нижнему регистру объекта Series используеться
series.str.lower()
stock['item'].str.lower()

# Поиск неявных дубликатов
# Неявные дубликаты ищут методом unique(). Он возвращает перечень уникальных значений в столбце:
tennis['name'].unique()

# Удаление неявных дубликатов
tennis['name'] = tennis['name'].replace('Roger Federer', 'Роджер Федерер')

duplicates = ['Roger Fderer', 'Roger Fdrer', 'Roger Federer'] # список неправильных имён
name = 'Роджер Федерер' # правильное имя
tennis['name'] = tennis['name'].replace(duplicates, name) # замена всех значений из duplicates на name
print(tennis) # датафрейм изменился, неявные дубликаты устранены 

# Изменение типов данных --------------------------
transactions['amount'] = pd.to_numeric(transactions['amount'], errors='coerce')
# errors='coerce'– что не сможет распарсить, заменит на NaN

transactions['amount'] =transactions['amount'].astype('int')

arrivals['target_datetime'] = pd.to_datetime(arrivals['target_time'], format='%Y-%m-%dZ%H:%M:%S')
# Методом to_datetime() превратим содержимое этого столбца в понятные для Python даты.
# Для этого строку форматируют, обращаясь к специальной системе обозначений, где:
# %d — день месяца (от 01 до 31)
# %m — номер месяца (от 01 до 12)
# %Y — четырёхзначный номер года (например, 2019)
# Z — стандартный разделитель даты и времени
# %H — номер часа в 24-часовом формате
# %I — номер часа в 12-часовом формате
# %M — минуты (от 00 до 59)
# %S — секунды (от 00 до 59)
# Пример:


# Метод to_datetime() работает и с форматом unix time. 
# Первый аргумент — это столбец со временем в формате unix time, второй аргумент unit со значением 's' сообщит о том, 
# что нужно перевести время в привычный формат с точностью до секунды.
# Часто приходится исследовать статистику по месяцам: например, узнать, на сколько минут сотрудник опаздывал в среднем. 
# Чтобы осуществить такой расчёт, нужно поместить время в класс DatetimeIndex и применить к нему атрибут month:
arrivals['month'] = pd.DatetimeIndex(arrivals['date_datetime']).month

# Функция для одной строки
clients['age_group'] = clients.apply(age_group_unemployed, axis=1) 




#-----------------------------------------------------------------------------------------------------
exoplanet.groupby('discovered').count()
exo_number = exoplanet.groupby('discovered')['radius'].count()


logs['source'].value_counts() # возвращает уникальные значения и количество их упоминаний
logs[logs['email'].isna()]
print(logs[logs['email'].isna()].head())


support_log_grouped['alert_group'] = support_log_grouped['user_id'].apply(alert_group)



# Задание 11. 
# Заполните пропуски в столбце `days_employed` медианными значениями по каждого типа занятости `income_type`.
# Решение из практикума
for t in data['income_type'].unique():
    data.loc[(data['income_type'] == t) & (data['total_income'].isna()), 'total_income'] = \
    data.loc[(data['income_type'] == t), 'total_income'].median()

# Моё решение
income_type_list = list(data['income_type'].unique())

for income_type in income_type_list:
    type_median = data[data['income_type'] == income_type]['total_income'].median()
    data[data['income_type'] == income_type] = \
    data[data['income_type'] == income_type].fillna({'total_income':type_median})
# По идеи, мое решение должно работать быстрее, но список типов можно собрать прямо в цикле,
# как это сделано в практикуме. И как видно из примера, использовать list() необязательно


# Рекурсия для извлечения списка из списков
def list_from_lists(main_list, list_to_append):
    for el in main_list:
        if isinstance(el, list):
            list_from_lists(el, list_to_append)
        else:
            list_to_append.append(el)


#---------------------- Примеры из вебинара ------------------------------------------------------
df.groupby('VIP')['transported'].mean()
df.groupby('VIP')['transported'].agg(['count', 'mean'])
df.groupby('VIP')['Age'].agg(['count', 'mean'])

df['age_group'] = pd.cut(df['age_group'] 5)
df.groupby('age_group')['transported'].mean().plot()
df.groupby('VIP')['age_group'].agg(['count', 'mean'])

df.groupby('VIP')['age_group'].mean().plot()
df.groupby('VIP')['age_group'].mean().plot(ylim=0)
df.groupby('VIP')['age_group'].mean().plot(kind='bar')
df.groupby('VIP')['age_group'].mean().plot.bar()
df.groupby('VIP')['age_group'].mean().sort_values(ascending=False).plot(kind='bar')

calls.groupby(['user_id', 'month']).agg(calls=('duration', 'count'))
calls.groupby(['user_id', 'month']).agg(minutes=('duration', 'sum'))
# ------------------------------------<Название> <Столбец>, <Что сделать>
# ------------------------------------<нового столбца>

# Если всё выражение взять в скобки, его части можно преносить по строкам
# не используя обратный слэш
(
    df.groupby('VIP')['age_group'].mean()
    .sort_values(ascending=False).plot(kind='bar')
    )

df['Cabin'].str[0]
df['Cabin'].str.replace(# ...
df['Cabin'].str.split('/')
df['Cabin'].str.split('/', maxsplit=1)
# maxsplit=-1 – отсутствие ограничение на количество изменений


def get_first_part(x):
    if not pd.isna(x): # если не пропуск
        return x.split('/')[0]
    return x # если пропуск

df['Cabin'].apply(get_first_part)

#--------- Статья "10 трюков библиотеки Python Pandas, которые вам нужны" -----------------------------
# https://proglib.io/p/pandas-tricks

# map
# Это классная команда для простого преобразования данных. 
# Определяете словарь, в котором «ключами» являются старые значения, а «значениями» – новые значения:
level_map = {1: 'high', 2: 'medium', 3: 'low'}
df['c_level'] = df['c'].map(level_map)
# Выбрать строки с конкретными идентификаторами
# В SQL используем SELECT * FROM… WHERE ID в («A001», «C022»,…) 
# и получаем записи с конкретными идентификаторами. 
# Если хотите сделать то же с помощью Python библиотеки Pandas, используйте
df_filter = df['ID'].isin(['A001','C022',...])
df[df_filter]

# --------------------- Простые визуализации ------------------------------------------------------------
df['time_spent'].hist(bins=100, range = (0, 1500))


# одна команда в несколько строк: не забыть заключить конструкцию в скобки 
(
    data
    .query('id == "3c1e4c52"')
    .plot(x='local_time', y='time_spent', 
          ylim=(0, 1000), style='o', grid=True, figsize=(12, 6))
)

(
    data.query('id == "3c1e4c52"')
    .pivot_table(index='date_hour', values='time_spent', aggfunc='median')
    .plot(grid=True, figsize=(12, 5))
) 


# --------------------- df.query(...) ------------------------------------------------------------

print(df.query('Is_Direct == True or Has_luggage == True')) 
df.query('color == @variable') # знаком @ обозначаеться внешняя переменная

good_ids  = too_fast_stat.query('too_fast  < 0.5')
good_data = data.query('id in @good_ids.index')

# Переименование столбцов при использовании сводных таблиц и образовании "мультииндекса"
id_name.columns = ['name', 'count']

# --------------------- df.merge(...) ------------------------------------------------------------

# В отличие от merge(), по умолчанию в join() установлен тип слияния how='left'
# Ещё методом join() можно объединять больше двух таблиц: 
# их набор передают списком вместо второго датафрейма.
df1.join(df2, on='a', rsuffix='_y')['c'] 

# --------------------- Коэффициент корреляции Пирсона ------------------------------------------------------------

df['height'].corr(df['weight'])
df['weight'].corr(df['height']) # поменяли местами рост и вес 
# 0.5196907833692264
# 0.5196907833692264 

hwa.corr()
#         height    weight       age      male
# height  1.000000  0.940822  0.683689  0.139229
# weight  0.940822  1.000000  0.678335  0.155443
# age     0.683689  0.678335  1.000000  0.005887
# male    0.139229  0.155443  0.005887  1.000000 

# --------------------- Матрица диаграмм рассеяния ------------------------------------------------------------

pd.plotting.scatter_matrix(df)
pd.plotting.scatter_matrix(df, figsize=(9, 9))

# --------------------- df.where() ------------------------------------------------------------
shopping.where(shopping != 'хамон', 'обойдусь')

big_nets_stat = final_stat.query('stations > 10')
station_stat_full['group_name'] = (
    station_stat_full['name']
    .where(station_stat_full['name'].isin(big_nets_stat.index), 'Другие')
)


# Метод .groupby возвращает "пары" объектов список уникальных ключей и Series значений к каждлому ключу

for developer_name, developer_data in IT_names.groupby('name'):
    print(
        'Имя {} встречается {} раза'.format(
            developer_name, len(developer_data)
        )
    ) 



spot_probs={k:spot_counts[k]/36 for k in spot_counts} # dictionary comprehension 

# Привести названия к нажнему регистру можно командой 
df.columns.str.lower(). 

# Для удобства можно еще сразу все данные вывести в процентах по пропущенным значениям. 
data.isna().sum()/len(data)*100` или `data.isna().mean()
# Возможно это будет более оптимально и позволит сразу увидеть где максимальное/минимальное 
# количество пропусков, и в каких колонках похожий процент пропусков.

# Техническую строку `<AxesSubplot:...>` можно убрать с помощью метода 
plt.show() 
# или добавления в конце кода `;`



