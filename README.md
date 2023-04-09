# Автоматическое построение плоской панорамы
В репозитории содержится код из статьи 
[https://habr.com/ru/articles/708986/](https://habr.com/ru/articles/708986/). 
Код реализует простой алгоритм для автоматического построения
плоского панорманого изображения из нескольких изображений.

# Запуск
Для запуска требуется `python3`

Usage: `main.py IMAGE_DIRECTORY_PATH OUTPUT_PATH`

Пример запуска на тестовых изображениях:
```commandline
python -m pip install -r requirements.txt
python main.py test_images/1 test_images/result_1.png
```
