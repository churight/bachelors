# Система розпізнавання жестової абетки

Проєкт призначений для збору власного датасету жестів, навчання моделей та розпізнавання жестів у реальному часі з вебкамери.

Підтримуються два типи моделей:

- **MLP** - працює з ключовими точками руки, які витягуються через MediaPipe.
- **CNN** - працює напряму із зображеннями жестів.

Основний спосіб роботи з проєктом - через графічний інтерфейс `joint_interface.py`.

## Вимоги

Потрібні Python та встановлені бібліотеки:

- `opencv-python`
- `mediapipe`
- `numpy`
- `pillow`
- `tensorflow`
- `scikit-learn`
- `h5py`

У цьому проєкті використовується віртуальне середовище:

```powershell
C:\Projects\venv310\Scripts\python.exe
```

## Структура проєкту

```text
Bachelors/
  Data/                    # Pickle-файли з ключовими точками
  Data/Photos/             # Фото-датасети
  models/                  # Навчені моделі та label map файли
  get_photos.py            # Збір фото з вебкамери
  extract_keypoints.py     # Витяг ключових точок руки
  train_models.py          # Навчання MLP/CNN моделей
  real_time_recogntion.py  # Розпізнавання у реальному часі
  joint_interface.py       # Графічний інтерфейс
```

## Запуск графічного інтерфейсу

З кореня проєкту:

```powershell
cd C:\Projects\Bachelors
C:\Projects\venv310\Scripts\python.exe joint_interface.py
```

В інтерфейсі доступні дії:

- **Create Dataset** - створити датасет з вебкамери.
- **Extract Keypoints** - витягнути ключові точки для MLP.
- **Train Model** - навчити власну модель.
- **Real-Time Recognition** - запустити розпізнавання жестів.
- **Stop Process** - зупинити поточний процес.

## Повний робочий сценарій

### 1. Створення датасету

1. У полі **Folder name** введіть назву датасету, наприклад `Test`.
2. Вкажіть кількість класів у полі **Classes**.
3. Вкажіть кількість зображень на клас у полі **Images per class**.
4. Натисніть **Create Dataset**.
5. Для кожного класу програма чекатиме натискання `Q`, після чого почне запис зображень.

Датасет буде збережено у:

```text
Data/Photos/<назва_датасету>/
```

Кожен клас зберігається в окремій папці:

```text
Data/Photos/Test/0/
Data/Photos/Test/1/
Data/Photos/Test/2/
...
```

### 2. Витяг ключових точок для MLP

Для навчання MLP потрібно спочатку отримати ключові точки руки.

1. Виберіть потрібний датасет у **Folder name**.
2. Натисніть **Extract Keypoints**.

Буде створено файл:

```text
Data/<назва_датасету>_keypoints.pickle
```

Наприклад:

```text
Data/Test_keypoints.pickle
```

### 3. Навчання власної моделі

У блоці **Real-Time Recognition Model** виберіть тип моделі:

- `MLP` - навчання на ключових точках.
- `CNN` - навчання на зображеннях.

У полі **Epochs** задайте кількість епох навчання.

Після цього натисніть **Train Model**.

Для MLP перед навчанням потрібно виконати **Extract Keypoints**.

Навчені моделі зберігаються в папку `models/`:

```text
models/custom_<назва_датасету>_mlp.h5
models/custom_<назва_датасету>_mlp_label_map.pkl

models/custom_<назва_датасету>_cnn.h5
models/custom_<назва_датасету>_cnn_label_map.pkl
```

Також створюється файл історії навчання:

```text
models/custom_<назва_датасету>_mlp.history.json
models/custom_<назва_датасету>_cnn.history.json
```

### 4. Запуск розпізнавання

Для запуску вбудованих моделей:

1. Виберіть `MLP` або `CNN`.
2. Для `MLP` виберіть одну з моделей: `K`, `O`, `R`, `Combined`.
3. Натисніть **Real-Time Recognition**.

Для запуску власної моделі:

1. Виберіть датасет у **Folder name**.
2. Виберіть тип моделі `MLP` або `CNN`.
3. У полі вибору моделі виберіть `Custom`.
4. Натисніть **Real-Time Recognition**.

У вікні розпізнавання модель тільки відображається як індикатор. Перемикання моделі всередині вікна розпізнавання вимкнено.

Керування у вікні розпізнавання:

- `Backspace` - видалити останній символ.
- `C` - очистити поточний текст.
- `Esc` - закрити вікно.

## Запуск через командний рядок

### Збір фото

```powershell
C:\Projects\venv310\Scripts\python.exe get_photos.py --data-dir Data/Photos/Test --classes 29 --dataset-size 100
```

### Витяг ключових точок

```powershell
C:\Projects\venv310\Scripts\python.exe extract_keypoints.py --data-dir Data/Photos/Test --output Data/Test_keypoints.pickle
```

### Навчання MLP

```powershell
C:\Projects\venv310\Scripts\python.exe train_models.py --data-dir Data/Photos/Test --keypoints Data/Test_keypoints.pickle --model-type MLP --name Test --epochs 30
```

### Навчання CNN

```powershell
C:\Projects\venv310\Scripts\python.exe train_models.py --data-dir Data/Photos/Test --model-type CNN --name Test --epochs 30
```

### Запуск стандартної MLP Combined

```powershell
C:\Projects\venv310\Scripts\python.exe real_time_recogntion.py
```

За замовчуванням запускається:

```text
MLP / Combined
```

### Запуск конкретної MLP моделі

```powershell
C:\Projects\venv310\Scripts\python.exe real_time_recogntion.py --model-type MLP --person K
```

Доступні варіанти:

```text
K
O
R
Combined
Custom
```

### Запуск власної MLP моделі

```powershell
C:\Projects\venv310\Scripts\python.exe real_time_recogntion.py --model-type MLP --person Custom --model-name Test --mlp-model-path models/custom_Test_mlp.h5 --mlp-labels-path models/custom_Test_mlp_label_map.pkl
```

### Запуск власної CNN моделі

```powershell
C:\Projects\venv310\Scripts\python.exe real_time_recogntion.py --model-type CNN --model-name Test --cnn-model-path models/custom_Test_cnn.h5 --cnn-labels-path models/custom_Test_cnn_label_map.pkl
```

## Примітки щодо MLP

MLP модель очікує один набір ключових точок руки, тобто 42 ознаки:

```text
21 точка * 2 координати = 42
```

Якщо під час витягу ключових точок MediaPipe знаходить дві руки, можуть з'явитися приклади з 84 ознаками. Під час навчання такі приклади автоматично пропускаються, якщо більшість датасету має довжину 42.

У логах це може виглядати так:

```text
Skipped 26 keypoint samples with non-42 feature length
```

Це нормальна поведінка.

## Типові проблеми

### Не відкривається камера

Перевірте, що:

- вебкамера підключена;
- камера не зайнята іншою програмою;
- Windows дозволяє Python доступ до камери.

### Для MLP пише, що немає keypoints-файлу

Перед навчанням MLP потрібно натиснути **Extract Keypoints**.

Очікуваний файл:

```text
Data/<назва_датасету>_keypoints.pickle
```

### Custom модель не запускається

Перевірте, що модель була навчена для поточного датасету і що в папці `models/` є відповідні файли:

```text
custom_<назва_датасету>_mlp.h5
custom_<назва_датасету>_mlp_label_map.pkl
```

або:

```text
custom_<назва_датасету>_cnn.h5
custom_<назва_датасету>_cnn_label_map.pkl
```

### Помилки сумісності Keras

У проєкті є сумісний завантажувач моделей для старіших `.h5` файлів і моделей, збережених новішими версіями Keras. Якщо виникає помилка з `batch_shape` або `DTypePolicy`, використовуйте запуск через актуальний `real_time_recogntion.py`, а не старі копії скрипта.

## Рекомендований порядок роботи

```text
1. Запустити joint_interface.py
2. Створити датасет
3. Для MLP виконати Extract Keypoints
4. Навчити модель через Train Model
5. Вибрати Custom
6. Запустити Real-Time Recognition
```
