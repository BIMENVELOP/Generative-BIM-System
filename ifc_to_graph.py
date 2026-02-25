"""
Прототип преобразования IFC-файла в графовую структуру для машинного обучения.
Версия: 0.1 (Proof of Concept)
Зависимости: ifcopenshell, networkx, numpy
"""

import ifcopenshell
import networkx as nx
import numpy as np
from collections import defaultdict
import json


class IFCToGraphConverter:
    """
    Конвертер IFC моделей в графовое представление.
    Узлы: строительные элементы (стены, плиты, окна, двери)
    Ребра: топологические и пространственные связи
    """
    
    def __init__(self, ifc_file_path):
        """
        Инициализация конвертера
        
        Args:
            ifc_file_path: путь к IFC файлу
        """
        self.ifc_file = ifcopenshell.open(ifc_file_path)
        self.graph = nx.MultiDiGraph()  # Мультиграф с направленными ребрами
        self.element_types = {
            'IfcWall': 'стена',
            'IfcWallStandardCase': 'стена',
            'IfcSlab': 'перекрытие',
            'IfcBeam': 'балка',
            'IfcColumn': 'колонна',
            'IfcWindow': 'окно',
            'IfcDoor': 'дверь',
            'IfcRoof': 'крыша',
            'IfcStair': 'лестница',
            'IfcRamp': 'пандус'
        }
        
    def extract_element_properties(self, element):
        """
        Извлечение свойств элемента IFC
        
        Args:
            element: элемент IFC
            
        Returns:
            dict: словарь с атрибутами элемента
        """
        properties = {
            'global_id': str(element.GlobalId),
            'name': str(element.Name) if element.Name else '',
            'type': self.element_types.get(element.is_a(), 'другое'),
            'ifc_class': element.is_a()
        }
        
        # Извлечение геометрических параметров (если доступны)
        try:
            if hasattr(element, 'Representation') and element.Representation:
                # Здесь можно добавить извлечение bounding box
                properties['has_geometry'] = True
            else:
                properties['has_geometry'] = False
        except:
            properties['has_geometry'] = False
            
        # Извлечение материалов (если доступны)
        try:
            if hasattr(element, 'HasAssociations'):
                for assoc in element.HasAssociations:
                    if assoc.is_a('IfcRelAssociatesMaterial'):
                        properties['material'] = str(assoc.RelatingMaterial.Name)
        except:
            pass
            
        return properties
    
    def find_spatial_connections(self, element1, element2, tolerance=0.1):
        """
        Определение типа связи между двумя элементами на основе пространственного анализа
        
        Args:
            element1: первый элемент
            element2: второй элемент
            tolerance: допуск для определения контакта (метры)
            
        Returns:
            str: тип связи или None
        """
        # Упрощенная проверка на основе bounding box
        # В реальном проекте здесь должен быть более сложный анализ геометрии
        
        try:
            # Получаем bounding box (упрощенно)
            bbox1 = self.get_bounding_box(element1)
            bbox2 = self.get_bounding_box(element2)
            
            if not bbox1 or not bbox2:
                return None
                
            # Проверка на вертикальную связь (опора)
            # Если элемент1 находится над элементом2 с небольшим зазором
            if abs(bbox1['z_min'] - bbox2['z_max']) < tolerance:
                # Проверка перекрытия по XY
                if (bbox1['x_min'] < bbox2['x_max'] and 
                    bbox1['x_max'] > bbox2['x_min'] and
                    bbox1['y_min'] < bbox2['y_max'] and 
                    bbox1['y_max'] > bbox2['y_min']):
                    return 'опора'
                    
            # Проверка на горизонтальное примыкание
            if (abs(bbox1['x_max'] - bbox2['x_min']) < tolerance or
                abs(bbox1['x_min'] - bbox2['x_max']) < tolerance):
                if (bbox1['y_min'] < bbox2['y_max'] and 
                    bbox1['y_max'] > bbox2['y_min']):
                    return 'примыкание'
                    
            # Проверка на пересечение (коллизию)
            if (bbox1['x_min'] < bbox2['x_max'] and 
                bbox1['x_max'] > bbox2['x_min'] and
                bbox1['y_min'] < bbox2['y_max'] and 
                bbox1['y_max'] > bbox2['y_min'] and
                bbox1['z_min'] < bbox2['z_max'] and 
                bbox1['z_max'] > bbox2['z_min']):
                return 'пересечение'
                
        except:
            pass
            
        return None
    
    def get_bounding_box(self, element):
        """
        Получение ограничивающего прямоугольника элемента
        (упрощенная реализация)
        """
        # В реальном проекте здесь должен быть парсинг геометрии IFC
        # Возвращаем заглушку для демонстрации
        return {
            'x_min': 0, 'x_max': 10,
            'y_min': 0, 'y_max': 10,
            'z_min': 0, 'z_max': 3
        }
    
    def build_graph(self):
        """
        Построение графа из IFC модели
        """
        print("Начало построения графа из IFC...")
        
        # Сбор всех элементов заданных типов
        elements = []
        for ifc_type in self.element_types.keys():
            try:
                elements.extend(self.ifc_file.by_type(ifc_type))
            except:
                continue
                
        print(f"Найдено элементов: {len(elements)}")
        
        # Добавление узлов в граф
        for element in elements:
            props = self.extract_element_properties(element)
            self.graph.add_node(
                props['global_id'],
                **props
            )
            
        # Поиск связей между элементами
        connection_count = 0
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements[i+1:], i+1):
                conn_type = self.find_spatial_connections(elem1, elem2)
                if conn_type:
                    self.graph.add_edge(
                        elem1.GlobalId,
                        elem2.GlobalId,
                        type=conn_type,
                        bidirectional=(conn_type in ['пересечение', 'примыкание'])
                    )
                    connection_count += 1
                    
        print(f"Найдено связей: {connection_count}")
        
        return self.graph
    
    def export_to_pytorch_geometric(self):
        """
        Экспорт графа в формат, совместимый с PyTorch Geometric
        """
        # Создание маппинга узлов в индексы
        nodes = list(self.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Матрица смежности
        edge_index = []
        edge_attr = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            # Кодируем тип связи в one-hot вектор
            if data['type'] == 'опора':
                edge_attr.append([1, 0, 0])
            elif data['type'] == 'примыкание':
                edge_attr.append([0, 1, 0])
            elif data['type'] == 'пересечение':
                edge_attr.append([0, 0, 1])
            else:
                edge_attr.append([0, 0, 0])
                
        # Атрибуты узлов
        node_features = []
        for node in nodes:
            node_data = self.graph.nodes[node]
            # Кодируем тип элемента в one-hot
            if node_data['type'] == 'стена':
                node_features.append([1, 0, 0, 0])
            elif node_data['type'] == 'перекрытие':
                node_features.append([0, 1, 0, 0])
            elif node_data['type'] == 'окно':
                node_features.append([0, 0, 1, 0])
            elif node_data['type'] == 'дверь':
                node_features.append([0, 0, 0, 1])
            else:
                node_features.append([0, 0, 0, 0])
                
        return {
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_index': np.array(edge_index, dtype=np.int64).T,  # PyG ожидает [2, num_edges]
            'edge_attr': np.array(edge_attr, dtype=np.float32),
            'node_ids': nodes
        }
    
    def save_graph_visualization(self, output_file='graph.html'):
        """
        Сохранение визуализации графа в HTML (для демонстрации)
        """
        import plotly.graph_objects as go
        
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        edge_traces = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            ))
            
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            text=[],
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[],
                colorbar=dict(
                    thickness=15,
                    title='Тип элемента'
                )
            )
        )
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (self.graph.nodes[node]['type'],)
            
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title='Граф BIM модели',
                           showlegend=False,
                           hovermode='closest'
                       ))
        
        fig.write_html(output_file)
        print(f"Визуализация сохранена в {output_file}")


# Пример использования
if __name__ == "__main__":
    # Путь к тестовому IFC файлу
    # В демо-версии можно использовать публичный пример
    import urllib.request
    import os
    
    test_ifc_url = "https://github.com/IfcOpenShell/IfcOpenShell/raw/master/test/input/wall-with-opening-and-window.ifc"
    test_file = "test_model.ifc"
    
    if not os.path.exists(test_file):
        print("Загрузка тестовой IFC модели...")
        urllib.request.urlretrieve(test_ifc_url, test_file)
    
    # Конвертация
    converter = IFCToGraphConverter(test_file)
    graph = converter.build_graph()
    
    # Экспорт для PyTorch Geometric
    pyg_data = converter.export_to_pytorch_geometric()
    
    print("\nСтатистика графа:")
    print(f"Узлов: {len(pyg_data['node_features'])}")
    print(f"Ребер: {pyg_data['edge_index'].shape[1]}")
    print(f"Размерность признаков узлов: {pyg_data['node_features'].shape[1]}")
    
    # Сохранение результатов
    with open('graph_data.json', 'w') as f:
        # Конвертируем numpy в списки для JSON
        json_data = {
            'node_features': pyg_data['node_features'].tolist(),
            'edge_index': pyg_data['edge_index'].tolist(),
            'edge_attr': pyg_data['edge_attr'].tolist()
        }
        json.dump(json_data, f, indent=2)
    
    print("Данные сохранены в graph_data.json")
    
    # Визуализация (опционально)
    try:
        converter.save_graph_visualization()
    except:
        print("Визуализация недоступна (plotly не установлен)")