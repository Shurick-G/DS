# --------------------- DecisionTreeClassifier() ------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(features, target) 

# Параметры умодели:
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
max_features=None, max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2, splitter='best')

answers = model.predict(new_features) 

# указываем случайное состояние (число)
model = DecisionTreeClassifier(random_state=12345)

# обучаем модель как раньше
model.fit(features, target)



import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('train_data.csv')

df.loc[df['last_price'] >  5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target) 


# В библиотеке sklearn метрики находятся в модуле sklearn.metrics. 
# Вычисляется accuracy функцией accuracy_score() (англ. «оценка правильности»).
from sklearn.metrics import accuracy_score 

# Функция принимает на вход два аргумента: 
# 1) правильные ответы, 2) предсказания модели. Возвращает она значение accuracy.
accuracy = accuracy_score(target, predictions) 




# ----- Пример -------
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# < импортируйте функцию расчёта accuracy из библиотеки sklearn >
from sklearn.metrics import accuracy_score 
df = pd.read_csv('/datasets/train_data.csv')
df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target)

test_df = pd.read_csv('/datasets/test_data.csv')

test_df.loc[test_df['last_price'] > 5650000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 5650000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']

train_predictions = model.predict(features)
test_predictions = model.predict(test_features)

print("Accuracy")
print("Обучающая выборка:", accuracy_score(target,      train_predictions))
print("Тестовая выборка:",  accuracy_score(test_target, test_predictions))


# ----- Деление на две выборки -------
from sklearn.model_selection import train_test_split 
df_train, df_valid = train_test_split(df, test_size=0.25, random_state=12345) 
# Напомним: в random_state мы могли записать всё что угодно, главное не None.



# --------------------- RandomForestClassifier() ------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

# Чтобы управлять количеством деревьев в лесу, пропишем гиперпараметр n_estimators 
# (от англ. number of estimators, «количество оценщиков»

model = RandomForestClassifier(random_state=12345, n_estimators=3)
model.fit(features, target)
model.predict(new_item)

# Правильность модели мы проверяли функцией accuracy_score(). 
# Но можно — и методом score(). Он считает accuracy для всех алгоритмов классификации.
model.score(features, target)

# --------------------- LogisticRegression() ------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
# Запишите модель в переменной, указав гиперпараметры. Для постоянства результата задайте random_state, равный 12345. 
# Добавьте дополнительные гиперпараметры: solver='lbfgs' и max_iter=1000. 
# Первый гиперпараметр позволяет выбрать алгоритм, который будет строить модель. 
# Алгоритм 'lbfgs' — один из самых распространённых. 
# Он подходит для большинства задач. Гиперпараметром max_iter задаётся максимальное количество итераций обучения. 
# Значение этого параметра по умолчанию равно 100, но в некоторых случаях понадобится больше итераций.
model = LogisticRegression(random_state=12345, solver='lbfgs', max_iter=1000)
model.fit(features, target)
model.predict(new_item)
model.score(features, target)


# --------------------- mean_squared_error()  MSE  -----------------------------------------------------------
from sklearn.metrics import mean_squared_error

answers = [623, 253, 150, 237]
predictions = [649, 253, 370, 148]

result = mean_squared_error(answers, predictions)



import pandas as pd
from sklearn.metrics import mean_squared_error
df = pd.read_csv('/datasets/train_data.csv')
features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000
# < найдите MSE >
predictions = pd.Series(target.mean(), index=target.index) 
print("MSE:", mse)

# --------------------- Дерево решений в регрессии  -----------------------------------------------------------
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/datasets/train_data.csv')

features = df.drop(['last_price'], axis=1)
target = df['last_price'] / 1000000

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345) # отделите 25% данных для валидационной выборки

best_model = None
best_result = 10000
best_depth = 0
for depth in range(1, 6):
    model = DecisionTreeRegressor(random_state=12345, max_depth=depth) # инициализируйте модель DecisionTreeRegressor с параметром random_state=12345 и max_depth=depth
    model.fit(features_train, target_train) # обучите модель на тренировочной выборке
    predictions_valid = model.predict(features_valid) # получите предсказания модели на валидационной выборке
    result =  mean_squared_error(target_valid, predictions_valid)**0.5 # посчитайте значение метрики rmse на валидационной выборке
    if result < best_result:
        best_model = model
        best_result = result
        best_depth = depth

print("RMSE наилучшей модели на валидационной выборке:", best_result, "Глубина дерева:", best_depth)



df.dtypes


# ---------------------------------------------- One-Hot Encoding  -----------------------------------------------------------

# Для прямого кодирования в библиотеке pandas есть функция 
pd.get_dummies(drop_first=False)


# Обучая логистическую регрессию, вы можете столкнуться с предупреждением библиотеки sklearn. Чтобы его отключить, 
# укажите аргумент solver='liblinear' (англ. solver «алгоритм решения»; library linear, «библиотека линейных алгоритмов»): 
model = LogisticRegression(solver='liblinear') 




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance.csv')

data_ohe = pd.get_dummies(data, drop_first=True)
target = data_ohe['Claim']
features = data_ohe.drop('Claim', axis=1)
# < напишите код здесь >
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

model =  LogisticRegression(random_state=12345, solver='liblinear', max_iter=1000)
model.fit(features_train, target_train)

print("Обучено!")

# ------------------------------  Ordinal Encoding (от англ. «кодирование по номеру категории»)  ---------------------------------------
# Чтобы выполнить кодирование, 
# в sklearn есть структура данных OrdinalEncoder (англ. «порядковый кодировщик»). 
# Она находится в модуле sklearn.preprocessing (от англ. «предобработка»). 

from sklearn.preprocessing import OrdinalEncoder 
# 1. Создаём объект этой структуры данных.
encoder = OrdinalEncoder() 
# 2. Чтобы получить список категориальных признаков, 
# вызываем метод fit() — как и в обучении модели. 
# Передаём ему данные как аргумент.
encoder.fit(data) 
# 3. Преобразуем данные функцией transform() (англ. «преобразовать»). 
# Изменённые данные будут храниться в переменной data_ordinal (англ. «порядковые данные»).
data_ordinal = encoder.transform(data) 
# Чтобы код добавил названия столбцов, оформим данные в структуру DataFrame():
data_ordinal = pd.DataFrame(encoder.transform(data), columns=data.columns)
# Если преобразование признаков требуется лишь один раз, как в нашей задаче, 
# код можно упростить вызовом функции fit_transform() (от англ. «подогнать и преобразовать»). 
# Она объединяет функции: fit() и transform(). 
data_ordinal = pd.DataFrame(encoder.fit_transform(data), columns=data.columns)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('/datasets/travel_insurance.csv')

encoder = OrdinalEncoder()
data_ordinal = pd.DataFrame(encoder.fit_transform(data), target = data_ordinal['Claim']

features = data_ordinal.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

# < напишите код здесь >
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

print("Обучено!")




# ------------------------------  Масштабирование признаков  ---------------------------------------
# ------------------------------  стандартизация данных  ---------------------------------------
# В sklearn есть отдельная структура для стандартизации данных — StandardScaler (от англ. «преобразователь масштаба методом стандартизации»). 
# Он находится в модуле sklearn.preprocessing. 
from sklearn.preprocessing import StandardScaler
# Создадим объект этой структуры и настроим его на обучающих данных. Настройка — это вычисление среднего и дисперсии:
scaler = StandardScaler()
scaler.fit(features_train) 
# Преобразуем обучающую и валидационную выборки функцией transform(). 
# Изменённые наборы сохраним в переменных: features_train_scaled (англ. «масштабированные признаки для обучения») 
# и features_valid_scaled (англ. «масштабированные признаки для проверки»):
features_train_scaled = scaler.transform(features_train)
features_valid_scaled = scaler.transform(features_valid)

# При записи изменённых признаков в исходный датафрейм код может вызывать предупреждение SettingWithCopy. 
# Причина в особенности поведения sklearn и pandas.  Специалисты уже привыкли игнорировать такое сообщение.
# Чтобы предупреждение не появлялось, в код добавляют строчку:
pd.options.mode.chained_assignment = None




import pandas as pd
from sklearn.model_selection import train_test_split
# < напишите код здесь >
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('travel_insurance.csv')

data_ohe = pd.get_dummies(data, drop_first=True)
target = data_ohe['Claim']
features = data_ohe.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

numeric = ['Duration', 'Net Sales', 'Commission (in value)', 'Age']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])

print(features_train.shape)


# ------------------------------  Метрики классификации  ---------------------------------------
# ------------------------------  Accuracy для решающего дерева  -------------------------------
accuracy_score()


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

# < напишите код здесь >
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

predicted_valid = model.predict(features_valid) # получите предсказания модели
accuracy_valid = accuracy_score(predicted_valid, target_valid)
print(accuracy_valid) 

# ------------------------------  Проверка адекватности модели  ---------------------------------------
# Чтобы оценить адекватность модели, проверим, как часто в целевом признаке встречается класс «1» или «0». 
# Количество уникальных значений подсчитывается методом value_counts(). Он группирует строго одинаковые величины
import pandas as pd
data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

# < напишите код здесь >
class_frequency = data['Claim'].value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar') 




import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

# чтобы работала функция value_counts(),
# мы преобразовали результат к pd.Series 
predicted_valid = pd.Series(model.predict(features_valid))

# < напишите код здесь >
class_frequency = predicted_valid.value_counts(normalize=True)
print(class_frequency)
class_frequency.plot(kind='bar')


# ------------------------------  Матрица ошибок  ---------------------------------------
# Матрица неточностей находится в знакомом модуле sklearn.metrics. 
# Функция confusion_matrix() принимает на вход верные ответы и предсказания, а возвращает матрицу ошибок.



# ------------------------------  Увеличение выборки  ---------------------------------------

answers = [0, 1, 0]
print(answers)
answers_x3 = answers * 3
print(answers_x3) 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

# < сделайте функцию из кода ниже >
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    return features_upsampled, target_upsampled 
    
# < добавьте перемешивание >
features_upsampled, target_upsampled = upsample(features_train, target_train, 10)
features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)

print(features_upsampled.shape)
print(target_upsampled.shape)


# ------------------------------  Уменьшение выборки  ---------------------------------------
# Преобразование проходит в несколько этапов:
# 1. Разделить обучающую выборку на отрицательные и положительные объекты;
# 2. Случайным образом отбросить часть из отрицательных объектов;
# 3. С учётом полученных данных создать новую обучающую выборку;
#    Перемешать данные. Положительные не должны идти следом за отрицательными: алгоритмам будет сложнее обучаться.

# Чтобы выбросить из таблицы случайные элементы, примените функцию sample(). 
# На вход она принимает аргумент frac (от англ. fraction, «доля»). 
# Возвращает случайные элементы в таком количестве, чтобы их доля от исходной таблицы была равна frac.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    # < напишите код здесь >
    features_downsampled = pd.concat([features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat([target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(features_train, target_train, 0.1)
features_downsampled, target_downsampled = shuffle(features_downsampled, target_downsampled, random_state=12345)

print(features_downsampled.shape)
print(target_downsampled.shape)

# ------------------------------ Изменение порога  ---------------------------------------
# В библиотеке sklearn вероятность классов вычисляет функция 
# predict_proba() (от англ. predict probabilities, «предсказать вероятности»). 
# На вход она получает признаки объектов, а возвращает вероятности:
probabilities = model.predict_proba(features)
# Для решающего дерева и случайного леса в sklearn тоже есть функция:
predict_proba()




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

# < напишите код здесь >
probabilities_valid  = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1] 

print(probabilities_one_valid[:5])




# ------------------------------ PR-кривая  ---------------------------------------
# ------------------------------ построение кривой --------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
precision, recall, thresholds = precision_recall_curve(target_valid, probabilities_valid[:, 1])

plt.figure(figsize=(6, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Кривая Precision-Recall')
plt.show()

# ------------------------------ ROC-кривая  ---------------------------------------
# Чтобы выявить, как сильно наша модель отличается от случайной, посчитаем площадь под 
# ROC-кривой — AUC-ROC (от англ. Area Under Curve ROC, «площадь под ROC-кривой»). 
# Это новая метрика качества, которая изменяется от 0 до 1. AUC-ROC случайной модели равна 0.5.
# Построить ROC-кривую поможет функция roc_curve() (англ. ROC-кривая) из модуля sklearn.metrics:
from sklearn.metrics import roc_curve
# На вход она принимает значения целевого признака и вероятности положительного класса. 
# Перебирает разные пороги и возвращает три списка: значения FPR, значения TPR и рассмотренные пороги.
fpr, tpr, thresholds = roc_curve(target, probabilities)



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# < напишите код здесь >
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# < напишите код здесь >
precision, recall, thresholds = precision_recall_curve(target_valid, probabilities_valid[:, 1])
fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid) 

plt.figure()

# < постройте график >

plt.plot(fpr, tpr)
# ROC-кривая случайной модели (выглядит как прямая)
plt.plot([0, 1], [0, 1], linestyle='--')

# < примените функции plt.xlim() и plt.ylim(), чтобы
#   установить границы осей от 0 до 1 >
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])

# < примените функции plt.xlabel() и plt.ylabel(), чтобы
#   подписать оси "False Positive Rate" и "True Positive Rate" >
plt.xlabel('False Positive Rate')
plt.ylabel('rue Positive Rate')

# < добавьте к графику заголовок "ROC-кривая" функцией plt.title() >
plt.title('ROC-кривая')

plt.show()

# ------------------------------ расчет AUC-ROC ---------------------------------------
from sklearn.metrics import roc_auc_score 



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# < напишите код здесь >
from sklearn.metrics import roc_auc_score 

data = pd.read_csv('/datasets/travel_insurance_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# < напишите код здесь >
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)


print(auc_roc)



# ------------------------------ Метрики регрессии ---------------------------------------
# -------------------------- Коэффициент детерминаци (R2)-------------------------------------
# Коэффициент детерминации, или метрика R2 (англ. coefficient of determination; R-squared), 
# вычисляет долю средней квадратичной ошибки модели от MSE среднего, а затем вычитает эту величину из единицы. 
# Увеличение метрики означает прирост качества модели. 
# Формула расчёта R2 выглядит так:
R2 = 1 - (MSE модели / MSE среднего)
# – Значение метрики R2 равно единице только в одном случае, если MSE нулевое. Такая модель предсказывает все ответы идеально.
# –  R2 равно нулю: модель работает так же, как и среднее.
# –  Если метрика R2 отрицательна, качество модели очень низкое.
# –  Значения R2 больше единицы быть не может.
from sklearn.metrics import r2_score
r2_score(target_valid, predicted_valid)


# __________________________________________________________________________________________________________________________
Максимизация R2: поиск модели
Время для практики.
Вы найдёте модель с наибольшим значением R2. Поэкспериментируйте в Jupyter Notebook и доведите эту метрику до 0.14.  
Алгоритм решения задачи:
Подготовьте библиотеки, данные и признаки — features и target. Разделите тестовую и обучающую выборки:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'] , axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345) 
Вычислите значение R2 функцией score().  R2 — метрика по умолчанию у моделей регрессии в sklearn:
model = LinearRegression()
model.fit(features_train, target_train)
print(model.score(features_valid, target_valid)) 
0.09710497146204988 
Лучший результат в этой задаче даёт алгоритм случайного леса. Найдите подходящие гиперпараметры: глубину дерева, число деревьев.
Чем больше деревьев, тем дольше учится модель. Поэтому сначала подберите глубину леса при небольшом числе деревьев:
for depth in range(1, 16, 1):
    model = RandomForestRegressor(n_estimators=20, max_depth=depth, random_state=12345)
    model.fit(features_train, target_train)
    # < напишите код здесь > 
Затем увеличивайте количество деревьев:
model = RandomForestRegressor(n_estimators=60, 
    max_depth=# < напишите код здесь >, random_state=12345)
model.fit(features_train, target_train)
print(model.score(features_train, target_train))
print(model.score(features_valid, target_valid)) 
Измерьте время обучения. Тренировка одной модели займёт секунды, а поиск оптимальных гиперпараметров в цикле — несколько минут.
В Jupyter Notebook время работы ячейки измеряет команда  %%time:
%%time

model = RandomForestRegressor(n_estimators=100, random_state=12345)
model.fit(features_train, target_train) 
CPU times: user 48 s, sys: 928 ms, total: 48.9 s
Wall time: 54.9 s 
Последняя строка Wall time (от англ. wall clock time, «время настенных часов») покажет время выполнения ячейки. 
В этом уроке нет проверки кода. Но чтобы решение прошло тесты, выполните два условия:
Не удаляйте первую ячейку тетради Jupyter.
Wall time обучения одной модели должно быть меньше 50 секунд.
Запомните алгоритм и гиперпараметры. Они пригодятся в следующем уроке, где вы обучите и проверите наилучшую модель.



# ------------------------------ Расчёт доверительного интервала --------------------------------------
# Упростить вычисления поможет распределение Стьюдента 
scipy.stats.t 
# В нём есть функция для доверительного интервала 
interval()
# , которая принимает на вход:
 # – alpha — уровень доверия, равный единице минус уровень значимости. 
 #   Правда, альфой в статистике принято называть сам уровень значимости, а уровень доверия — 
 #   β. Так что разработчки SciPy выбрали для этого параметра не самое удачное название.
 # – df (от англ. degrees of freedom) — число степеней свободы, равное n - 1
# – loc (от англ. location) — среднее распределение, равное оценке среднего. Для выборки sample вычисляется так: sample.mean().
# – scale (англ. «масштаб») — стандартное отклонение распределения, равное оценке стандартной ошибки. Вычисляется так: sample.sem().import pandas as pd
from scipy import stats as st

sample = pd.Series([
    439, 518, 452, 505, 493, 470, 498, 442, 497, 
    423, 524, 442, 459, 452, 463, 488, 497, 500,
    476, 501, 456, 425, 438, 435, 516, 453, 505, 
    441, 477, 469, 497, 502, 442, 449, 465, 429,
    442, 472, 466, 431, 490, 475, 447, 435, 482, 
    434, 525, 510, 494, 493, 495, 499, 455, 464,
    509, 432, 476, 438, 512, 423, 428, 499, 492, 
    493, 467, 493, 468, 420, 513, 427])

print("Cреднее:", sample.mean())

confidence_interval = st.t.interval(0.95, len(sample)-1, sample.mean(), sample.sem())# < напишите код здесь >



# ------------------------------ Кросс-валидация в sklearn --------------------------------------
# Оценить качество модели кросс-валидацией позволяет функция 
cross_val_score() #(от англ. cross validation score, «оценка скользящего контроля») 
# из модуля sklearn.model_selection (от англ. «выбор модели»).

# Пример
from sklearn.model_selection import cross_val_score
cross_val_score(model, features, target, cv=5)

# На вход функция принимает несколько аргументов. Например:
# model — модель для кросс-валидации. 
# Она обучается в ходе перекрёстной проверки, поэтому на вход подаётся необученной. 
# Допустим, для дерева решений нужна такая модель:

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
# features — признаки;
# target — целевой признак;
# cv (от англ. cross validation) — число блоков для кросс-валидации, по умолчанию их пять.
# Делить данные на блоки или валидационную и обучающую выборки функция не требует. 
# Все эти процедуры происходят автоматически. На выходе от каждой валидации получаем список оценок качества моделей. 
# Каждая оценка равна model.score() на валидацинной выборке. Например, для задачи классификации — это accuracy.