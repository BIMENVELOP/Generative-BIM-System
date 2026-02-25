python

"""
Прототип гибридной нейросетевой архитектуры (GNN + CNN) для генеративного проектирования.
Версия: 0.1 (Proof of Concept)
Зависимости: torch, torch-geometric, torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    Слой графового внимания для обработки топологии BIM
    """
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.norm = nn.LayerNorm(out_channels * heads)
        
    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        x = self.norm(x)
        return F.elu(x)


class CNNEncoder(nn.Module):
    """
    CNN энкодер для обработки геометрических признаков (как в U-Net)
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            # [B, 3, 64, 64] -> [B, 64, 32, 32]
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # [B, 64, 32, 32] -> [B, 128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # [B, 128, 16, 16] -> [B, 256, 8, 8]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # [B, 256, 8, 8] -> [B, 512, 4, 4]
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Проекция в латентное пространство
        self.projection = nn.Linear(512 * 4 * 4, latent_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        latent = self.projection(features)
        return latent


class CrossModalAttention(nn.Module):
    """
    Механизм cross-modal внимания между графовыми и визуальными признаками
    """
    def __init__(self, graph_dim, cnn_dim, hidden_dim=128):
        super().__init__()
        
        # Проекции для механизма внимания
        self.graph_query = nn.Linear(graph_dim, hidden_dim)
        self.cnn_key = nn.Linear(cnn_dim, hidden_dim)
        self.cnn_value = nn.Linear(cnn_dim, hidden_dim)
        
        # Выходная проекция
        self.output_proj = nn.Linear(hidden_dim, graph_dim)
        
    def forward(self, graph_features, cnn_features):
        """
        Args:
            graph_features: [num_nodes, graph_dim] - признаки узлов
            cnn_features: [batch_size, cnn_dim] - глобальные признаки от CNN
        """
        # Проецируем в общее пространство
        q = self.graph_query(graph_features)  # [num_nodes, hidden]
        
        # Для простоты используем один и тот же CNN признак для всех узлов
        # В реальной модели здесь будет более сложный механизм
        k = self.cnn_key(cnn_features).unsqueeze(1)  # [batch, 1, hidden]
        v = self.cnn_value(cnn_features).unsqueeze(1)  # [batch, 1, hidden]
        
        # Вычисляем внимание
        attn_weights = torch.matmul(q.unsqueeze(0), k.transpose(-2, -1))  # [batch, num_nodes, 1]
        attn_weights = F.softmax(attn_weights / np.sqrt(q.size(-1)), dim=-1)
        
        # Применяем внимание
        attended = torch.matmul(attn_weights.transpose(-2, -1), v)  # [batch, 1, hidden]
        
        # Обновляем признаки узлов
        updated_features = q + attended.squeeze(1)  # residual connection
        
        return self.output_proj(updated_features)


class HybridGenerator(nn.Module):
    """
    Гибридный генератор (GNN + CNN) для создания планировок
    """
    def __init__(self, 
                 node_feature_dim=4,    # one-hot тип элемента
                 graph_hidden_dim=128,
                 cnn_latent_dim=256,
                 num_gnn_layers=3):
        super().__init__()
        
        # Начальная проекция признаков узлов
        self.node_embedding = nn.Linear(node_feature_dim, graph_hidden_dim)
        
        # Слои GNN
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(
                GraphAttentionLayer(graph_hidden_dim, graph_hidden_dim)
            )
            
        # CNN энкодер для геометрии
        self.cnn_encoder = CNNEncoder(in_channels=3, latent_dim=cnn_latent_dim)
        
        # Механизм cross-modal внимания
        self.cross_attention = CrossModalAttention(
            graph_dim=graph_hidden_dim,
            cnn_dim=cnn_latent_dim
        )
        
        # Декодер для генерации новых атрибутов узлов
        self.node_decoder = nn.Sequential(
            nn.Linear(graph_hidden_dim, graph_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim * 2, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, node_feature_dim),
            nn.Sigmoid()  # Для бинарных one-hot признаков
        )
        
    def forward(self, graph_data, condition_image):
        """
        Args:
            graph_data: объект PyG Data с полями x (node features), edge_index
            condition_image: тензор изображения [B, C, H, W] с условием
        """
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        
        # 1. Обработка графа
        x = self.node_embedding(x)
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            
        # 2. Обработка изображения CNN
        cnn_features = self.cnn_encoder(condition_image)  # [B, cnn_dim]
        
        # 3. Cross-modal внимание
        x = self.cross_attention(x, cnn_features)
        
        # 4. Декодирование новых признаков узлов
        new_node_features = self.node_decoder(x)
        
        return new_node_features


class HybridDiscriminator(nn.Module):
    """
    Дискриминатор для оценки реалистичности и структурной целостности
    """
    def __init__(self, node_feature_dim=4, graph_hidden_dim=64):
        super().__init__()
        
        # Дискриминатор для графовой структуры
        self.graph_disc = nn.Sequential(
            GraphAttentionLayer(node_feature_dim, graph_hidden_dim),
            GraphAttentionLayer(graph_hidden_dim, graph_hidden_dim),
            global_mean_pool,  # Пуллинг графа
            nn.Linear(graph_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Патч-дискриминатор для изображений (PatchGAN)
        self.patch_disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        )
        
    def forward(self, graph_data, image):
        """
        Возвращает две оценки: структурную и визуальную
        """
        # Оценка структуры графа
        graph_score = self.graph_disc(graph_data.x, graph_data.edge_index, graph_data.batch)
        
        # Оценка качества изображения
        image_score = self.patch_disc(image)
        
        return graph_score, image_score


# Демонстрация работы
if __name__ == "__main__":
    print("Инициализация гибридной модели...")
    
    # Параметры модели
    node_feature_dim = 4  # one-hot для типов элементов
    batch_size = 2
    num_nodes = 10
    
    # Создание синтетических данных для демонстрации
    # Графовые данные
    x = torch.randn(batch_size * num_nodes, node_feature_dim)
    edge_index = torch.randint(0, batch_size * num_nodes, (2, 20))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    graph_data = Data(x=x, edge_index=edge_index)
    graph_data.batch = batch
    
    # Изображение условия
    condition_image = torch.randn(batch_size, 3, 64, 64)
    
    # Создание моделей
    generator = HybridGenerator(node_feature_dim=node_feature_dim)
    discriminator = HybridDiscriminator(node_feature_dim=node_feature_dim)
    
    # Прямой проход
    print("\nПрямой проход через генератор:")
    with torch.no_grad():
        generated_nodes = generator(graph_data, condition_image)
        print(f"Вход: граф с {graph_data.x.shape[0]} узлами")
        print(f"Выход генератора: {generated_nodes.shape} (новые признаки узлов)")
        
        graph_score, image_score = discriminator(graph_data, condition_image)
        print(f"\nОценка дискриминатора:")
        print(f"- Структурная оценка: {graph_score.mean().item():.4f}")
        print(f"- Визуальная оценка: {image_score.mean().item():.4f}")
    
    print("\n✓ Модель успешно инициализирована и протестирована")
    print("  (Демонстрация на синтетических данных)")