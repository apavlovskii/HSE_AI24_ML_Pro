from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import joblib
import re


app = FastAPI()

# Предварительно поместим файл модели в текущий каталог
MODEL_PATH = "ridge_model.pickle"

# Загрузим модель из файла
model = joblib.load(MODEL_PATH)


def parse_torque(torque):
    """
    Парсим различные варианты представления значений torque.

    Args:
    torque (str): Torque в различных форматах.

    Returns:
    tuple: (torque_value, rpm) - значение torque, средний rpm из диапазона
    """
    if not isinstance(torque, str) or not torque.strip():
        return (None, None)

    # Убираем разделитель разрядов
    torque = torque.replace(',', '')
    # Достаем значение torque и единицу измерения (Nm или kgm)
    torque_match = re.search(r'([\d.]+)\s*(Nm|kgm)', torque, re.IGNORECASE)
    if torque_match:
        torque_value = float(torque_match.group(1))
        unit = torque_match.group(2).lower()
    else:
        # Обрабатываем вариант, когда kgm приходит в скобках
        torque_match = re.search(r'([\d.]+)@\s*[\d,]*(?:-\d+)?\((kgm)@.*\)', torque, re.IGNORECASE)
        if torque_match:
            torque_value = float(torque_match.group(1))
            unit = torque_match.group(2).lower()
        else:
            return (None, None)

    # Переводим kgm в Nm
    if unit == 'kgm':
        torque_value *= 9.8  # 1 kgm ≈ 9.8 Nm

    # Округлим до 1 знака
    torque_value = round(torque_value, 1)

    # Достаем диапазон rpm или единственное значение
    rpm_match = re.search(r'(\d{1,5}(?:,\d{3})?)(?:[,-]\s*(\d{1,5}(?:,\d{3})?))?\s*rpm', torque, re.IGNORECASE)
    if rpm_match:
        min_rpm = int(rpm_match.group(1))
        max_rpm = int(rpm_match.group(2)) if rpm_match.group(2) else min_rpm
    else:
        # Обрабатываем формат (kgm@ rpm) в скобках
        match = re.search(r'\s([\d,]+(?:-[\d,]+)?)\s*\(kgm@ rpm\)', torque, re.IGNORECASE)
        if not match:
            min_rpm = None
            max_rpm = None
        else:
            values = match.group(1)
            # Если в значение содержится - то обрабатываем как диапазон
            if '-' in values:
                min_rpm, max_rpm = map(float, values.split('-'))
            else:  # иначе считаем min = max
                min_rpm = float(values)
                max_rpm = min_rpm

    # В кач-ве rpm берем среднее значение из диапазона
    rpm = int((min_rpm + max_rpm)//2) if min_rpm is not None else None

    return (torque_value, rpm)


# Опишем структуру данных при помощи Pydantic
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


# Сюда поместим необходимые действия по предобработке данных
def preprocess_data(item_data: List[dict]) -> pd.DataFrame:
    """
    Предобработка входных данных.

    Args:
    Список объектов в формате словаря (json-style)

    Returns:
    Pandas DataFrame
    """

    # Преобразуем список объектов в DataFrame
    df = pd.DataFrame(item_data)

    X = df.drop(columns=["selling_price"])

    # Предобработаем числовые колонки
    X["mileage"] = X["mileage"].str.extract(r"(\d+\.\d+|\d+)").astype(float)
    X["engine"] = X["engine"].str.extract(r"(\d+\.\d+|\d+)").astype(float)
    X["max_power"] = X["max_power"].str.extract(r"(\d+\.\d+|\d+)").astype(float)
    X[["torque", "rpm"]] = X["torque"].apply(lambda x: pd.Series(parse_torque(x)))
    X["seats"] = X["seats"].astype(str)

    return X


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Предсказать цену одного автомобиля
    """
    # Вызовем предобработку
    input_data = preprocess_data([item.dict()])
    # Выполним предсказание
    prediction = model.predict(input_data)[0]

    return prediction


@app.post("/predict_items")
def predict_items(file: UploadFile) -> str:
    """
    Предсказать цену автомобилей, переданных в виде структуры в csv файле
    """
    # Прочитаем входной файл
    contents = file.file.read()
    test_data = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Вызовем предобработку
    input_data = preprocess_data(test_data.to_dict(orient="records"))

    # Выполним предсказание
    predictions = model.predict(input_data)
    # Добавим колонку с предсказаниями в исходный датасет
    test_data["Predicted_Price"] = predictions

    # Вернем датасет обратно в виде csv
    output = io.StringIO()
    test_data.to_csv(output, index=False)
    output.seek(0)

    return output.getvalue()
