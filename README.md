markdown

# Prototype: Generative BIM System

Данный репозиторий содержит прототипы кода, демонстрирующие ключевые технологические решения проекта:

- `ifc_to_graph.py` - конвертер IFC моделей в графовое представление для машинного обучения
- `hybrid_model_demo.py` - прототип гибридной архитектуры GNN + U-Net с механизмом внимания

## Быстрый старт

```bash
# Установка зависимостей
pip install ifcopenshell networkx torch torch-geometric torchvision plotly

# Запуск конвертера IFC -> Graph
python ifc_to_graph.py

# Запуск демо гибридной модели
python hybrid_model_demo.py