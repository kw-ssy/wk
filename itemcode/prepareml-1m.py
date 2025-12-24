import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import chardet  # 用于自动检测文件编码


# 自动检测文件编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))  # 读取前10KB来检测编码
    return result['encoding']


# 读取 movies.dat 文件，增强编码容错能力
def read_movies_file(file_path):
    # 尝试检测文件编码
    try:
        encoding = detect_encoding(file_path)
        print(f"检测到文件编码: {encoding}")
    except:
        encoding = 'utf-8'
        print("编码检测失败，默认使用 utf-8")

    movies = []
    try:
        # 使用检测到的编码打开文件，并设置错误处理为'replace'
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                # 使用 '::' 分割（符合 MovieLens 格式）
                parts = line.split('::')
                if len(parts) < 3:
                    continue
                # 提取电影ID、标题和类型
                movie_id = parts[0]
                title = parts[1]
                genres = parts[2]
                # 组合标题和类型作为描述信息
                description = f"{title} ({genres})"
                movies.append(description)
    except UnicodeDecodeError:
        print("使用检测的编码失败，尝试使用 Latin-1 编码")
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('::')
                    if len(parts) < 3:
                        continue
                    movie_id = parts[0]
                    title = parts[1]
                    genres = parts[2]
                    description = f"{title} ({genres})"
                    movies.append(description)
        except Exception as e:
            print(f"无法读取文件: {e}")
            return []

    print(f"成功读取 {len(movies)} 部电影")
    return movies


# 加载预训练模型（支持多语言，处理特殊字符）
tokenizer = BertTokenizer.from_pretrained('./pretrain')  # 保持原路径
model = BertModel.from_pretrained('./pretrain').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# 修改为电影数据文件路径
movies_file_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/movies_filtered.dat'
movie_descriptions = read_movies_file(movies_file_path)

# 生成BERT特征
features = []
print("开始编码电影描述...")
for description in tqdm(movie_descriptions, desc="Encoding Progress"):
    # 处理描述中的空格和特殊字符
    encoded_input = tokenizer(
        description,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(model.device)

    with torch.no_grad():
        output = model(**encoded_input)

    # 平均池化获取特征
    feature = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    features.append(feature)

# 归一化特征到 [-1, 1]
features = np.array(features)
min_val = np.min(features)
max_val = np.max(features)
normalized_features = -1 + 2 * (features - min_val) / (max_val - min_val) if (
                                                                                     max_val - min_val) != 0 else np.zeros_like(
    features)

# 修改为电影特征保存路径
output_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/text_feat_movies.npy'
np.save(output_path, normalized_features)
print(f"特征已保存至 {output_path}，形状：{normalized_features.shape}")

import json
import os


def convert_ratings_to_json(input_file, output_file):
    """将电影评分数据(ratings.dat)转换为用户-电影交互JSON"""
    if not os.path.exists(input_file):
        print(f"错误: 输入文件{input_file}不存在")
        return

    interactions = {}
    try:
        # 检测文件编码（处理可能的编码问题）
        import chardet
        with open(input_file, 'rb') as f:
            encoding = chardet.detect(f.read(10000))['encoding'] or 'utf-8'
        print(f"检测到文件编码: {encoding}")

        with open(input_file, 'r', encoding=encoding, errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 电影数据使用::分隔（userID::movieID::rating::timestamp）
                parts = line.split('::')
                if len(parts) < 2:
                    continue  # 跳过格式错误的行

                user_id = parts[0]
                movie_id = parts[1]
                # 确保movie_id为整数（电影ID通常是数字）
                if movie_id.isdigit():
                    movie_id = int(movie_id)

                # 按用户ID分组，收集交互的电影ID
                if user_id not in interactions:
                    interactions[user_id] = []
                # 避免重复添加同一部电影
                if movie_id not in interactions[user_id]:
                    interactions[user_id].append(movie_id)

        # 写入JSON，确保列表元素为数字时无引号
        with open(output_file, 'w', encoding='utf-8') as jf:
            json_str = '{\n'
            total_users = len(interactions)
            for i, (user, movies) in enumerate(interactions.items()):
                # 用户ID保持字符串格式（带引号）
                movies_str = ', '.join(map(str, movies))  # 数字直接转为字符串，无引号
                json_str += f'  "{user}": [{movies_str}]'
                # 处理逗号分隔
                if i < total_users - 1:
                    json_str += ',\n'
                else:
                    json_str += '\n'
            json_str += '}'
            jf.write(json_str)

        print(f"成功转换{total_users}个用户的交互记录，保存至{output_file}")

    except Exception as e:
        print(f"转换出错: {e}")


if __name__ == "__main__":
    # 电影数据文件路径配置
    input_file = "/data/cclsol/wk/LLMRec-main1/data/ml-1m/ratings.dat"  # 电影评分数据
    output_file = "/data/cclsol/wk/LLMRec-main1/data/ml-1m/user_movie_interactions.json"  # 输出文件

    # 执行转换
    convert_ratings_to_json(input_file, output_file)

import json
import random


def split_data(json_file_path, train_ratio=0.8):
    """
    将JSON文件中的用户-物品交互数据划分为训练集和测试集
    :param json_file_path: JSON文件路径
    :param train_ratio: 训练集所占比例，默认为0.8
    :return: 训练集和测试集
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    train_data = {}
    test_data = {}

    for user_id, item_list in data.items():
        random.shuffle(item_list)
        split_index = int(len(item_list) * train_ratio)
        train_data[user_id] = item_list[:split_index]
        test_data[user_id] = item_list[split_index:]

    return train_data, test_data


def save_to_json(data, file_path):
    """
    将数据保存为JSON文件
    :param data: 要保存的数据（字典形式）
    :param file_path: JSON文件路径
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# 示例使用
json_file_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/user_movie_interactions.json'  # 请替换为你的JSON文件路径
train_data, test_data = split_data(json_file_path)

# 保存训练集为JSON文件
train_file_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/train电影训练集.json'
save_to_json(train_data, train_file_path)

# 保存测试集为JSON文件
test_file_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/test电影测试集.json'
save_to_json(test_data, test_file_path)

print(f"训练集已保存至 {train_file_path}")
print(f"测试集已保存至 {test_file_path}")


import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
import os

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = os.path.join(path, 'train电影训练集.json')
        val_file = os.path.join(path, 'val电影验证集.json')
        test_file = os.path.join(path, 'test电影测试集.json')

        # 新增：用户ID映射表
        self.user_id_mapping = {}  # 原始用户ID到连续ID的映射
        self.reverse_user_id_mapping = {}  # 连续ID到原始用户ID的反向映射
        self.item_id_mapping = {}  # 物品ID到连续ID的映射
        self.reverse_item_id_mapping = {}  # 连续物品ID到原始ID的反向映射

        self.n_users = 0  # 连续用户ID总数
        self.n_items = 0  # 连续物品ID总数
        self.n_train = 0
        self.n_test = 0
        self.n_val = 0
        self.neg_pools = {}
        self.exist_users = []  # 存储连续用户ID

        # 读取并构建用户和物品ID映射
        self._build_id_mapping(train_file, val_file, test_file)

        # 加载物品文本特征（假设特征数与物品数一致）
        text_feats_path = os.path.join(path, 'text_feat_movies.npy')
        self.text_feats = np.load(text_feats_path)
        self.n_items = self.text_feats.shape[0]  # 确保物品数与特征数一致

        # 构建交互矩阵和数据集
        self._build_interaction_matrix(train_file, val_file, test_file)
        self.print_statistics()

    def _build_id_mapping(self, train_file, val_file, test_file):
        """构建用户和物品的连续ID映射"""
        all_users = set()
        all_items = set()

        # 从训练集收集用户和物品
        with open(train_file, 'r') as f:
            train_data = json.load(f)
            for uid_str, items in train_data.items():
                uid = int(uid_str)
                all_users.add(uid)
                all_items.update(items)

        # 从验证集收集用户和物品
        with open(val_file, 'r') as f:
            val_data = json.load(f)
            for uid_str, items in val_data.items():
                uid = int(uid_str)
                all_users.add(uid)
                all_items.update(items)

        # 从测试集收集用户和物品
        with open(test_file, 'r') as f:
            test_data = json.load(f)
            test_data = {k: v for k, v in test_data.items() if v}  # 过滤空列表
            for uid_str, items in test_data.items():
                uid = int(uid_str)
                all_users.add(uid)
                all_items.update(items)

        # 生成用户连续ID映射
        self.user_id_mapping = {uid: idx for idx, uid in enumerate(sorted(all_users))}
        self.reverse_user_id_mapping = {idx: uid for uid, idx in self.user_id_mapping.items()}
        self.n_users = len(self.user_id_mapping)
        self.exist_users = list(self.user_id_mapping.values())  # 存储原始用户ID（可根据需要调整）

        # 生成物品连续ID映射
        self.item_id_mapping = {item: idx for idx, item in enumerate(sorted(all_items))}
        self.reverse_item_id_mapping = {idx: item for item, idx in self.item_id_mapping.items()}

    def _build_interaction_matrix(self, train_file, val_file, test_file):
        """构建用户-物品交互矩阵和数据集"""
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.train_items = {}
        self.test_set = {}
        self.val_set = {}

        # 处理训练集
        with open(train_file, 'r') as f:
            train_data = json.load(f)
            for uid_str, items in train_data.items():
                if not items:
                    continue
                uid = int(uid_str)
                if uid not in self.user_id_mapping:
                    continue  # 跳过未出现过的用户（理论上不应出现）
                mapped_uid = self.user_id_mapping[uid]
                mapped_items = []
                for item in items:
                    if item not in self.item_id_mapping:
                        continue  # 跳过未出现过的物品
                    mapped_item = self.item_id_mapping[item]
                    self.R[mapped_uid, mapped_item] = 1.
                    mapped_items.append(mapped_item)
                self.train_items[mapped_uid] = mapped_items
                self.n_train += len(mapped_items)

        # 处理测试集
        with open(test_file, 'r') as f:
            test_data = json.load(f)
            for uid_str, items in test_data.items():
                if not items:
                    continue
                uid = int(uid_str)
                if uid not in self.user_id_mapping:
                    continue
                mapped_uid = self.user_id_mapping[uid]
                mapped_items = [self.item_id_mapping[item] for item in items if item in self.item_id_mapping]
                if mapped_items:
                    self.test_set[mapped_uid] = mapped_items
                    self.n_test += len(mapped_items)

        # 处理验证集
        with open(val_file, 'r') as f:
            val_data = json.load(f)
            for uid_str, items in val_data.items():
                if not items:
                    continue
                uid = int(uid_str)
                if uid not in self.user_id_mapping:
                    continue
                mapped_uid = self.user_id_mapping[uid]
                mapped_items = [self.item_id_mapping[item] for item in items if item in self.item_id_mapping]
                if mapped_items:
                    self.val_set[mapped_uid] = mapped_items
                    self.n_val += len(mapped_items)

    def get_adj_mat(self):
        """获取邻接矩阵（保持原有逻辑，使用连续ID）"""
        try:
            t1 = time()
            adj_mat = sp.load_npz(os.path.join(self.path, 's_adj_mat电影.npz'))
            norm_adj_mat = sp.load_npz(os.path.join(self.path, 's_norm_adj_mat电影.npz'))
            mean_adj_mat = sp.load_npz(os.path.join(self.path, 's_mean_adj_mat电影.npz'))
            print('已加载邻接矩阵', adj_mat.shape, time() - t1)
        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(os.path.join(self.path, 's_adj_mat电影.npz'), adj_mat)
            sp.save_npz(os.path.join(self.path, 's_norm_adj_mat电影.npz'), norm_adj_mat)
            sp.save_npz(os.path.join(self.path, 's_mean_adj_mat电影.npz'), mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        """创建邻接矩阵（使用连续用户和物品ID）"""
        t1 = time()
        total_nodes = self.n_users + self.n_items
        adj_mat = sp.dok_matrix((total_nodes, total_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        # 将交互矩阵R转换为CSR格式以提高计算效率
        R_csr = self.R.tocsr()
        adj_mat[:self.n_users, self.n_users:] = R_csr
        adj_mat[self.n_users:, :self.n_users] = R_csr.T
        adj_mat = adj_mat.todok()
        print('已创建邻接矩阵', adj_mat.shape, time() - t1)

        t2 = time()
        norm_adj_mat, mean_adj_mat = self._normalize_adj(adj_mat)
        print('已归一化邻接矩阵', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def _normalize_adj(self, adj):
        """归一化邻接矩阵"""
        def normalize(adj_matrix):
            rowsum = np.array(adj_matrix.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv.dot(adj_matrix).tocoo()

        # 带自环的归一化
        adj_with_self_loop = adj + sp.eye(adj.shape[0])
        norm_adj = normalize(adj_with_self_loop)
        # 不带自环的归一化
        mean_adj = normalize(adj)
        return norm_adj, mean_adj

    def sample(self):
        """采样用户-正负样本（使用连续用户ID）"""
        if self.batch_size <= self.n_users:
            users = rd.sample(list(self.user_id_mapping.values()), self.batch_size)  # 使用原始用户ID采样
        else:
            users = [rd.choice(list(self.user_id_mapping.values())) for _ in range(self.batch_size)]

        mapped_users = [self.user_id_mapping[uid] for uid in users]  # 转换为连续用户ID
        pos_items, neg_items = [], []

        for mu in mapped_users:
            pos_items += self._sample_pos_items(mu)
            neg_items += self._sample_neg_items(mu)

        return mapped_users, pos_items, neg_items

    def _sample_pos_items(self, u):
        """采样正样本"""
        pos_items = self.train_items.get(u, [])
        return [np.random.choice(pos_items)] if pos_items else [0]  # 避免空列表

    def _sample_neg_items(self, u):
        """采样负样本"""
        pos_items = set(self.train_items.get(u, []))
        neg_items = [i for i in range(self.n_items) if i not in pos_items]
        return [np.random.choice(neg_items)] if neg_items else [0]  # 避免空列表

    def print_statistics(self):
        """打印数据统计信息"""
        total_interactions = self.n_train + self.n_test + self.n_val
        sparsity = total_interactions / (self.n_users * self.n_items) if (self.n_users * self.n_items) != 0 else 0
        print(f"用户数: {self.n_users}, 物品数: {self.n_items}")
        print(f"训练交互数: {self.n_train}, 验证交互数: {self.n_val}, 测试交互数: {self.n_test}")
        print(f"总交互数: {total_interactions}, 稀疏度: {sparsity:.6f}")


if __name__ == "__main__":
    path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m'
    batch_size = 1024
    data_obj = Data(path, batch_size)
    adj_mat, norm_adj_mat, mean_adj_mat = data_obj.get_adj_mat()


import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix


def json_to_csr(json_data):
    # 收集所有用户ID和物品ID
    user_ids = sorted(json_data.keys(), key=int)
    item_ids = set()
    for items in json_data.values():
        item_ids.update(items)
    item_ids = sorted(item_ids)

    # 创建用户和物品的索引映射
    user_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}

    # 准备CSR矩阵所需的数据
    data = []
    row_indices = []
    col_indices = []

    # 填充数据
    for user_id, items in json_data.items():
        user_idx = user_to_idx[user_id]
        for item_id in items:
            item_idx = item_to_idx[item_id]
            data.append(1.0)  # 交互值为1.0
            row_indices.append(user_idx)
            col_indices.append(item_idx)

    # 创建CSR矩阵
    num_users = len(user_ids)
    num_items = len(item_ids)
    csr = csr_matrix((data, (row_indices, col_indices)), shape=(num_users, num_items), dtype='float32')

    return csr


def main():
    # 从文件读取JSON数据
    json_file = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/user_movie_interactions.json'  # 修改为你的JSON文件路径
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到JSON文件 '{json_file}'")
        return
    except json.JSONDecodeError:
        print(f"错误：JSON文件格式不正确")
        return

    # 转换为CSR矩阵
    csr_matrix_result = json_to_csr(json_data)

    # 保存为pkl文件
    output_file = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/train_mat电影'
    with open(output_file, 'wb') as file:
        pickle.dump(csr_matrix_result, file)

    print(f"CSR矩阵已保存到 {output_file}")
    print(f"矩阵形状: {csr_matrix_result.shape}")
    print(f"非零元素数量: {csr_matrix_result.nnz}")


if __name__ == "__main__":
    main()

#
import numpy as np
import pickle
from collections import defaultdict


def generate_user_samples(data_path, output_path, random_seed=42):
    """
    从MovieLens格式的ratings.dat中为每个用户生成正样本和负样本

    参数:
    data_path (str): ratings.dat文件的路径
    output_path (str): 保存样本的pkl文件路径
    random_seed (int): 随机数生成器的种子
    """
    np.random.seed(random_seed)

    # 步骤1: 读取数据并构建用户-电影评分字典
    user_movies = defaultdict(list)
    all_movies = set()

    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 处理MovieLens格式: userID::movieID::rating::timestamp
            parts = line.strip().split('::')
            if len(parts) >= 3:
                user_id = int(parts[0])
                movie_id = int(parts[1])
                rating = float(parts[2])

                user_movies[user_id].append((movie_id, rating))
                all_movies.add(movie_id)

    # 步骤2: 为每个用户选择正样本和负样本
    user_samples = {}

    for user_id, movies in user_movies.items():
        # 正样本: 评分最高的电影(如果有多个最高分，选择第一个)
        positive_movie = max(movies, key=lambda x: x[1])[0]

        # 该用户已评分的所有电影
        rated_movies = set([movie for movie, _ in movies])

        # 负样本: 随机选择一个用户未评分过的电影
        unrated_movies = list(all_movies - rated_movies)
        if unrated_movies:
            negative_movie = np.random.choice(unrated_movies)
            # 确保使用Python原生整数类型
            user_samples[user_id] = {0: int(positive_movie), 1: int(negative_movie)}
        else:
            print(f"警告: 用户 {user_id} 已对所有电影评分，跳过...")

    # 步骤3: 保存结果到pkl文件
    with open(output_path, 'wb') as file:
        pickle.dump(user_samples, file)

    print(f"样本生成完成，已保存到 {output_path}")
    print(f"处理的用户数: {len(user_samples)}")


# 使用示例
if __name__ == "__main__":
    # 请替换为实际文件路径
    data_file = "/data/cclsol/wk/LLMRec-main1/data/ml-1m/ratings.dat"
    output_file = "/data/cclsol/wk/LLMRec-main1/data/ml-1m/augmented_user_movie_samples"

    generate_user_samples(data_file, output_file)

import json
import os
import chardet  # 用于自动检测文件编码


def detect_encoding(file_path):
    """自动检测文件编码格式"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取部分数据用于检测
    return chardet.detect(raw_data)['encoding']


def filter_movies_by_interactions(interactions_file, movies_file, output_file):
    # 读取用户交互数据中的电影ID
    with open(interactions_file, 'r', encoding='utf-8') as f:
        interactions_data = json.load(f)

    # 提取所有唯一的电影ID（转为字符串，避免类型不匹配）
    movie_ids = set()

    # 遍历JSON中的每个用户及其交互的电影列表
    for user_id, movie_list in interactions_data.items():
        for movie_id in movie_list:
            movie_ids.add(str(movie_id))  # 将电影ID转为字符串

    print(f"从交互数据中提取了 {len(movie_ids)} 个唯一电影ID")

    # 自动检测 movies.dat 的编码
    movies_encoding = detect_encoding(movies_file)
    print(f"检测到 movies.dat 的编码为: {movies_encoding}")

    # 读取并过滤电影数据（使用检测到的编码）
    filtered_lines = []
    try:
        with open(movies_file, 'r', encoding=movies_encoding, errors='replace') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 1 and parts[0] in movie_ids:
                    filtered_lines.append(line)
    except UnicodeDecodeError as e:
        print(f"使用 {movies_encoding} 编码读取失败，尝试 fallback 到 latin-1")
        # 极端情况：使用 latin-1 编码（能兼容所有字节）
        with open(movies_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 1 and parts[0] in movie_ids:
                    filtered_lines.append(line)

    print(f"从 {movies_file} 中保留了 {len(filtered_lines)} 条记录")

    # 写入过滤后的文件（使用 UTF-8 编码输出，避免后续读取问题）
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)

    print(f"过滤后的文件已保存至 {output_file}")


if __name__ == "__main__":
    # 配置文件路径（请根据实际路径修改）
    base_dir = '/data/cclsol/wk/LLMRec-main1/data/ml-1m' # 你的数据目录
    interactions_file = os.path.join(base_dir, "user_movie_interactions.json")
    movies_file = os.path.join(base_dir, "movies.dat")
    output_file = os.path.join(base_dir, "movies_filtered.dat")  # 过滤后的输出文件

    filter_movies_by_interactions(interactions_file, movies_file, output_file)

import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm
import os

# 加载两个768维度的Sentence-BERT模型
model1 = SentenceTransformer('./pretrain')
model2 = SentenceTransformer('./pretrain1')


def load_dict_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def process_batch_data(batch_data):
    batch_texts = []
    batch_keys = []
    for key, value in batch_data.items():
        all_texts = []
        for v in value.values():
            if isinstance(v, list):
                if all(isinstance(item, list) for item in v):
                    flat_list = [str(item) for sublist in v for item in sublist]
                    all_texts.append(' '.join(flat_list))
                else:
                    all_texts.append(' '.join(map(str, v)))
            else:
                all_texts.append(str(v))
        text = ' '.join(all_texts)
        batch_texts.append(text)
        batch_keys.append(key)
    return batch_texts, batch_keys


def encode_and_normalize(batch_texts, model1, model2):
    # 使用两个模型分别进行编码
    embeddings1 = model1.encode(batch_texts, convert_to_numpy=True)
    embeddings2 = model2.encode(batch_texts, convert_to_numpy=True)

    # 拼接两个编码结果
    combined_embeddings = np.concatenate((embeddings1, embeddings2), axis=1)

    # 将数据类型转换为 float64
    combined_embeddings = combined_embeddings.astype(np.float64)

    # 线性归一化到 -0.1 到 0.1 范围
    min_val = np.min(combined_embeddings, axis=0, keepdims=True)
    max_val = np.max(combined_embeddings, axis=0, keepdims=True)

    # 处理所有元素相同的情况
    scale = max_val - min_val
    scale[scale == 0] = 1.0

    normalized_embeddings = -0.1 + 0.2 * (combined_embeddings - min_val) / scale

    return normalized_embeddings


def align_user_embeddings(user_profiles, target_user_ids, default_embedding=None):
    """
    对齐用户嵌入，确保与目标用户ID列表一致

    参数:
    user_profiles: 原始用户画像字典 {user_id: embedding}
    target_user_ids: 目标用户ID列表
    default_embedding: 缺失用户的默认嵌入值

    返回:
    对齐后的用户嵌入字典
    """
    aligned_embeddings = {}

    # 如果没有提供默认嵌入，创建一个全零的默认嵌入
    if default_embedding is None:
        # 假设嵌入维度是1536 (768*2)
        default_embedding = np.zeros(1536, dtype=np.float64)

    # 为每个目标用户ID分配嵌入
    for user_id in target_user_ids:
        if user_id in user_profiles:
            aligned_embeddings[user_id] = user_profiles[user_id]
        else:
            aligned_embeddings[user_id] = default_embedding

    return aligned_embeddings


def main():
    data_file_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/user_profiles_dictml-1m'
    output_file = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/user_profiles_embedding_dict'
    ui_graph_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/train_mat电影'
    batch_size = 100

    # 加载用户画像数据
    print(f"加载用户画像数据: {data_file_path}")
    user_profiles = load_dict_data(data_file_path)

    # 加载UI图数据以获取目标用户ID列表
    print(f"加载UI图数据: {ui_graph_path}")
    try:
        with open(ui_graph_path, 'rb') as f:
            ui_graph = pickle.load(f)
        # 获取用户ID列表（假设UI图的行数是用户数量）
        target_user_ids = list(range(ui_graph.shape[0]))  # 假设用户ID是连续的0到n-1
        print(f"目标用户数量: {len(target_user_ids)}")
    except Exception as e:
        print(f"无法加载UI图数据: {e}")
        print("使用默认用户ID列表（连续整数）")
        target_user_ids = list(range(1892))  # 默认使用69878个连续用户ID

    # 编码用户画像
    print("开始编码用户画像...")
    encoded_data = {}
    total_batches = len(user_profiles) // batch_size + (1 if len(user_profiles) % batch_size != 0 else 0)

    for start in tqdm(range(0, len(user_profiles), batch_size), total=total_batches, desc="编码进度"):
        end = start + batch_size
        batch_data = dict(list(user_profiles.items())[start:end])

        batch_texts, batch_keys = process_batch_data(batch_data)
        batch_embeddings = encode_and_normalize(batch_texts, model1, model2)

        # 将嵌入与用户ID关联
        for i, key in enumerate(batch_keys):
            encoded_data[key] = batch_embeddings[i]

    print(f"编码完成，原始用户嵌入数量: {len(encoded_data)}")

    # 对齐用户嵌入
    print(f"开始对齐用户嵌入到目标长度: {len(target_user_ids)}")
    aligned_embeddings = align_user_embeddings(encoded_data, target_user_ids)

    print(f"对齐后用户嵌入数量: {len(aligned_embeddings)}")

    # 保存编码结果
    with open(output_file, 'wb') as f:
        pickle.dump(aligned_embeddings, f)

    print(f"编码结果已保存到 {output_file}")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm
import chardet  # 用于自动检测文件编码


def detect_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read(100000))['encoding']


def generate_movie_embeddings():
    # 加载两个Sentence-BERT模型（保持原路径）
    model1 = SentenceTransformer('./pretrain')
    model2 = SentenceTransformer('./pretrain1')

    # --------------------------
    # 1. 读取电影基础数据（movies.dat）
    # --------------------------
    movies_dat_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/movies_filtered.dat'
    movies_data = {
        "title": [],  # 电影标题
        "movie_id": []  # 电影ID（用于对齐增强数据）
    }

    # 检测文件编码
    encoding = detect_encoding(movies_dat_path) or 'utf-8'
    print(f"movies.dat编码: {encoding}")

    with open(movies_dat_path, 'r', encoding=encoding, errors='replace') as file:
        for line in file:
            parts = line.strip().split('::')
            if len(parts) >= 2:
                movie_id = int(parts[0])
                # 提取标题（去除年份）
                title_with_year = parts[1]
                if '(' in title_with_year and ')' in title_with_year:
                    title = title_with_year[:title_with_year.rfind('(')].strip()
                else:
                    title = title_with_year.strip()
                movies_data["movie_id"].append(movie_id)
                movies_data["title"].append(title)

    # --------------------------
    # 2. 读取电影增强属性数据（之前生成的pickle文件）
    # --------------------------
    augmented_dict_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/augmented_attribute_dictml-1m属性'
    with open(augmented_dict_path, 'rb') as f:
        augmented_dict = pickle.load(f)
    print(f"加载增强属性数据，共{len(augmented_dict)}条记录")

    # --------------------------
    # 3. 合并基础数据与增强属性
    # --------------------------
    combined_data = {
        "title": [],  # 电影标题
        "director": [],  # 导演（增强属性）
        "country": [],  # 国家（增强属性）
        "language": [],  # 语言（增强属性）
        "starring": [],  # 主演（增强属性）
        "popularity": []  # 受欢迎度（增强属性）
    }

    # 以movies.dat的电影ID为基准对齐数据
    for movie_id, title in zip(movies_data["movie_id"], movies_data["title"]):
        combined_data["title"].append(title)
        # 从增强属性中提取对应字段，无数据则用空字符串填充
        if movie_id in augmented_dict:
            attr = augmented_dict[movie_id]
            combined_data["director"].append(attr.get('director', ""))
            combined_data["country"].append(attr.get('country', ""))
            combined_data["language"].append(attr.get('language', ""))
            combined_data["starring"].append(attr.get('starring', ""))
            combined_data["popularity"].append(attr.get('popularity', ""))
        else:
            # 增强属性中无该电影数据
            combined_data["director"].append("")
            combined_data["country"].append("")
            combined_data["language"].append("")
            combined_data["starring"].append("")
            combined_data["popularity"].append("")

    # --------------------------
    # 4. 对所有属性进行编码（两个模型特征拼接）
    # --------------------------
    batch_size = 100  # 批次处理大小
    encoded_data = {}  # 存储编码结果

    # 对每个属性字段分别编码
    for key, values in tqdm(combined_data.items(), desc="总编码进度"):
        encoded_list = []
        num_values = len(values)
        num_batches = (num_values // batch_size) + (1 if num_values % batch_size != 0 else 0)

        # 批次处理避免内存占用过大
        for batch_idx in tqdm(range(num_batches), desc=f"编码字段: {key}", leave=False):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_values = values[start:end]

            # 转换为字符串并处理空值
            batch_str = [str(v) if v else "" for v in batch_values]

            # 两个模型分别编码
            embeddings1 = model1.encode(batch_str, convert_to_numpy=True)
            embeddings2 = model2.encode(batch_str, convert_to_numpy=True)

            # 拼接特征并归一化到[-0.1, 0.1]
            for emb1, emb2 in zip(embeddings1, embeddings2):
                combined_emb = np.concatenate((emb1, emb2))  # 768 + 768 = 1536维
                combined_emb = combined_emb.astype(np.float64)

                # 归一化
                min_val = np.min(combined_emb)
                max_val = np.max(combined_emb)
                if max_val - min_val == 0:
                    normalized_emb = np.full_like(combined_emb, 0.0)
                else:
                    normalized_emb = -0.1 + 0.2 * (combined_emb - min_val) / (max_val - min_val)

                encoded_list.append(normalized_emb)

        encoded_data[key] = encoded_list

    # --------------------------
    # 5. 验证编码范围并保存结果
    # --------------------------
    # 检查是否所有向量都在[-0.1, 0.1]范围内
    for key, emb_list in encoded_data.items():
        for idx, emb in enumerate(emb_list):
            if np.any(emb < -0.1) or np.any(emb > 0.1):
                print(f"警告: {key}第{idx}个向量超出范围，最大值{np.max(emb)}，最小值{np.min(emb)}")

    # 保存编码结果
    output_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/movie_attribute_embedding_dict'
    with open(output_path, 'wb') as f:
        pickle.dump(encoded_data, f)
    print(f"电影属性编码已保存至{output_path}，包含{len(encoded_data)}个字段")
    print(f"每个字段编码维度: {len(encoded_data['title'][0])}（若数据非空）")


if __name__ == "__main__":
    generate_movie_embeddings()

#
import pickle
import numpy as np
import chardet


def detect_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read(100000))['encoding']


# 读取用户属性 pkl 文件（电影用户画像）
try:
    user_profiles_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/user_profiles_dictml-1m'
    with open(user_profiles_path, 'rb') as f:
        user_attributes = pickle.load(f)
    print(f"加载用户属性数据，共{len(user_attributes)}个用户")
except FileNotFoundError:
    print("User attributes file not found. Please check the path.")
    exit(1)
except Exception as e:
    print(f"Error reading user attributes: {e}")
    exit(1)

# 读取用户-电影交互记录 pkl 文件（CSR矩阵）
try:
    interactions_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/train_mat电影'
    with open(interactions_path, 'rb') as f:
        user_item_interactions = pickle.load(f)
    print(f"加载交互数据，形状: {user_item_interactions.shape}")
except FileNotFoundError:
    print("User-item interactions file not found.")
    exit(1)
except Exception as e:
    print(f"Error reading interactions: {e}")
    exit(1)

# 读取电影属性 pkl 文件（增强属性）
try:
    movie_attr_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/augmented_attribute_dictml-1m属性'
    with open(movie_attr_path, 'rb') as f:
        item_attributes = pickle.load(f)
    print(f"加载电影属性数据，共{len(item_attributes)}部电影")
except FileNotFoundError:
    print("Movie attributes file not found.")
    exit(1)
except Exception as e:
    print(f"Error reading movie attributes: {e}")
    exit(1)

# 初始化知识图谱三元组列表
kg_triples = []

# --------------------------
# 1. 处理用户属性（生成用户相关三元组）
# --------------------------
for user_id, attributes in user_attributes.items():
    user_entity = f"User_{user_id}"  # 统一格式：User_123

    # 映射用户属性到关系
    for attr_name, attr_value in attributes.items():
        # 跳过空值
        if not attr_value or attr_value.strip() == "":
            continue

        # 定义属性对应的关系
        if attr_name == 'age':
            relation = "Age is"
        elif attr_name == 'gender':
            relation = "Gender is"
        elif attr_name == 'liked_genre':
            relation = "Liked genre is"
        elif attr_name == 'disliked_genre':
            relation = "Disliked genre is"
        elif attr_name == 'country':
            relation = "Country is"
        elif attr_name == 'language':
            relation = "Language is"
        else:
            continue  # 跳过未定义的属性

        # 添加三元组 (用户实体, 关系, 属性值)
        kg_triples.append((user_entity, relation, attr_value.strip()))

# --------------------------
# 2. 处理用户-电影交互（生成交互关系三元组）
# --------------------------
rows, cols = user_item_interactions.nonzero()
print(f"处理交互记录，共{len(rows)}条非零交互")

for row, col in zip(rows, cols):
    user_id = row  # 行索引对应用户ID
    movie_id = col  # 列索引对应电影ID
    user_entity = f"User_{user_id}"
    movie_entity = f"Movie_{movie_id}"  # 统一格式：Movie_456
    relation = "Has watched"  # 电影领域更贴切的关系描述
    kg_triples.append((user_entity, relation, movie_entity))

# --------------------------
# 3. 处理电影属性（生成电影相关三元组）
# --------------------------
for movie_id, attributes in item_attributes.items():
    movie_entity = f"Movie_{movie_id}"  # 统一格式：Movie_456

    # 处理导演
    director = attributes.get('director', '').strip()
    if director:
        kg_triples.append((movie_entity, "Directed by", director))

    # 处理国家
    country = attributes.get('country', '').strip()
    if country:
        kg_triples.append((movie_entity, "Produced in", country))

    # 处理语言
    language = attributes.get('language', '').strip()
    if language:
        kg_triples.append((movie_entity, "Language is", language))

    # 处理主演（可能有多个，拆分后添加）
    starring = attributes.get('starring', '').strip()
    if starring:
        # 拆分多个主演（按逗号或分号）
        actors = [a.strip() for a in starring.replace(';', ',').split(',') if a.strip()]
        for actor in actors:
            kg_triples.append((movie_entity, "Starring", actor))

    # 处理受欢迎度
    popularity = attributes.get('popularity', '').strip()
    if popularity:
        kg_triples.append((movie_entity, "Popularity is", popularity))

# --------------------------
# 4. 保存知识图谱到文件
# --------------------------
output_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/combined_knowledge_graph电影.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    for triple in kg_triples:
        # 用制表符分隔三元组
        f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")

print(f"知识图谱构建完成，共{len(kg_triples)}个三元组")
print(f"保存路径: {output_path}")


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm
import os
from typing import List, Dict, Any, Tuple


# 配置参数
class Config:
    BATCH_SIZE = 100
    MAX_SEQ_LENGTH = 256
    EMBEDDING_DIM = 768 * 2
    NORMALIZE_RANGE = (-0.1, 0.1)
    CHUNK_SIZE = 512
    OVERLAP_SIZE = 64
    OUTPUT_DIR = '/data/cclsol/wk/LLMRec-main1/data/netflix/'
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'augmented_attribute_embedding_dict新电影')


# 文本分块函数
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    if not text:
        return [""]
    tokens = text.split()
    if len(tokens) <= chunk_size:
        return [text]
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i: i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks


# 归一化函数
def normalize_embedding(embedding: np.ndarray, min_val: float = -0.1, max_val: float = 0.1) -> np.ndarray:
    if np.all(embedding == 0):
        return embedding
    norm = np.linalg.norm(embedding)
    if norm < 1e-6:
        embedding += np.random.normal(0, 1e-7, embedding.shape)
        norm = np.linalg.norm(embedding)
    normalized = embedding / norm
    scaled = min_val + (max_val - min_val) * (normalized + 1) / 2
    return scaled


# 编码长文本
def encode_long_text(text: str, model1: SentenceTransformer, model2: SentenceTransformer, config: Config) -> np.ndarray:
    chunks = chunk_text(text, config.CHUNK_SIZE, config.OVERLAP_SIZE)
    chunk_embeddings = []
    for chunk in chunks:
        embedding1 = model1.encode(chunk, convert_to_numpy=True, normalize_embeddings=True)
        embedding2 = model2.encode(chunk, convert_to_numpy=True, normalize_embeddings=True)
        combined = np.concatenate((embedding1, embedding2))
        chunk_embeddings.append(combined)
    if not chunk_embeddings:
        return np.zeros(config.EMBEDDING_DIM)
    weights = np.ones(len(chunk_embeddings))
    if len(weights) > 2:
        weights[0] = 1.5
        weights[-1] = 1.5
    aggregated = np.average(chunk_embeddings, axis=0, weights=weights)
    return aggregated


# 主函数（修复数据合并逻辑）
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print("加载模型中...")
    model1 = SentenceTransformer('./pretrain')
    model2 = SentenceTransformer('./pretrain1')
    model1.max_seq_length = Config.MAX_SEQ_LENGTH
    model2.max_seq_length = Config.MAX_SEQ_LENGTH

    # 读取艺术家数据（包含真实artist_id）
    print("读取艺术家数据...")
    artists_dat_path = '/data/cclsol/wk/LLMRec-main1/data/netflix/item_attribute.csv'
    artists_data = {"id": [], "title": []}
    with open(artists_dat_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                artists_data["id"].append(parts[0])  # 提取真实artist_id
                artists_data["title"].append(parts[1])

    num_artists = len(artists_data["title"])
    print(f"读取到 {num_artists} 条记录")

    # 读取pickle字典文件
    print("读取属性字典...")
    dict_path = '/data/cclsol/wk/LLMRec-main1/data/netflix/movie_profiles.pkl'
    with open(dict_path, 'rb') as f:
        dict_data = pickle.load(f)

    # 合并数据（使用真实artist_id）
    dict_data = {str(k): v for k, v in dict_data.items()}
    print("合并数据...")
    combined_data = {
        "title": [],
        "summarization": [],
        "reasoning": []
    }

    for i in tqdm(range(num_artists), desc="合并数据"):
        artist_id = artists_data["id"][i]  # 使用真实artist_id
        combined_data["title"].append(artists_data["title"][i])

        if artist_id in dict_data:
            item = dict_data[artist_id]
            combined_data["summarization"].append(item.get('summarization', ""))
            combined_data["reasoning"].append(item.get('reasoning', ""))
        else:
            combined_data["summarization"].append("")
            combined_data["reasoning"].append("")

    # 验证合并后的数据多样性
    print("\n验证合并后的数据多样性:")
    for attr in ["summarization", "reasoning"]:
        values = combined_data[attr]
        unique_count = len(set(values[:100]))
        print(f"{attr} 的前100条记录中唯一值数量: {unique_count}/100")

    # 分批次编码数据
    print("\n开始编码数据...")
    encoded_data = {key: [] for key in combined_data}

    for key in combined_data:
        print(f"编码属性: {key}")
        values = combined_data[key]
        num_batches = (len(values) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE

        # 初始化空列表存储所有批次的结果
        all_embeddings = []

        for batch_idx in tqdm(range(num_batches), desc=f"批次 {key}"):
            start = batch_idx * Config.BATCH_SIZE
            end = start + Config.BATCH_SIZE
            batch_values = values[start:end]

            batch_embeddings = []
            for value in batch_values:
                embedding = encode_long_text(str(value), model1, model2, Config)
                normalized = normalize_embedding(embedding, *Config.NORMALIZE_RANGE)
                batch_embeddings.append(normalized)

            # 将当前批次的结果添加到总结果列表中
            all_embeddings.extend(batch_embeddings)

            # 检查编码多样性
            if batch_idx == 0 and len(batch_embeddings) >= 2:
                sim = np.dot(batch_embeddings[0], batch_embeddings[1]) / (
                        np.linalg.norm(batch_embeddings[0]) * np.linalg.norm(batch_embeddings[1])
                )
                print(f"  第一批编码相似度: {sim:.4f}")

        # 将所有批次的结果保存到encoded_data
        encoded_data[key] = all_embeddings

    # 保存编码结果
    print(f"\n保存编码结果到 {Config.OUTPUT_FILE}")
    with open(Config.OUTPUT_FILE, 'wb') as f:
        pickle.dump(encoded_data, f)

    print("完成!")


if __name__ == "__main__":
    main()

#
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import pickle
# from tqdm import tqdm
# import os
# from typing import List, Dict, Any, Tuple
#
#
# # 配置类
# class Config:
#     BATCH_SIZE = 32  # 降低批次大小，避免内存溢出
#     MAX_SEQ_LENGTH = 256  # 模型最大输入长度
#     EMBEDDING_DIM = 768 * 2  # 两个模型拼接后的维度
#     NORMALIZE_RANGE = (-0.1, 0.1)  # 归一化范围
#     CHUNK_SIZE = 512  # 长文本分块大小
#     OVERLAP_SIZE = 64  # 分块重叠大小
#
#
# # 文本处理工具函数
# def process_text(value: Any) -> str:
#     """将各种类型的值转换为文本"""
#     if isinstance(value, list):
#         if all(isinstance(item, list) for item in value):
#             # 嵌套列表展平
#             return ' '.join([str(item) for sublist in value for item in sublist])
#         else:
#             return ' '.join(map(str, value))
#     return str(value)
#
#
# def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
#     """将长文本分割为重叠的块"""
#     if not text or len(text) < chunk_size:
#         return [text]
#
#     tokens = text.split()
#     chunks = []
#     for i in range(0, len(tokens), chunk_size - overlap):
#         chunks.append(' '.join(tokens[i:i + chunk_size]))
#     return chunks
#
#
# # 改进的归一化方法
# def normalize_embeddings(embeddings: np.ndarray, min_val: float = -0.1, max_val: float = 0.1) -> np.ndarray:
#     """对批量嵌入进行归一化，处理全零向量和相同值的情况"""
#     # 检查是否为单向量
#     if embeddings.ndim == 1:
#         embeddings = embeddings.reshape(1, -1)
#
#     # 计算每列的统计信息
#     col_min = np.min(embeddings, axis=0, keepdims=True)
#     col_max = np.max(embeddings, axis=0, keepdims=True)
#     col_range = col_max - col_min
#
#     # 处理全零列
#     zero_mask = (col_range < 1e-8)
#     if np.any(zero_mask):
#         # 对全零列使用单位向量
#         embeddings[:, zero_mask.squeeze()] = np.random.normal(0, 1e-6, size=(embeddings.shape[0], np.sum(zero_mask)))
#         col_min = np.min(embeddings, axis=0, keepdims=True)
#         col_max = np.max(embeddings, axis=0, keepdims=True)
#         col_range = col_max - col_min
#
#     # 归一化到指定范围
#     normalized = min_val + (max_val - min_val) * (embeddings - col_min) / col_range
#     return normalized
#
#
# # 编码长文本
# def encode_long_text(
#         text: str,
#         model1: SentenceTransformer,
#         model2: SentenceTransformer,
#         config: Config
# ) -> np.ndarray:
#     """对长文本进行分块编码并聚合结果"""
#     chunks = chunk_text(text, config.CHUNK_SIZE, config.OVERLAP_SIZE)
#
#     if not chunks or not chunks[0]:
#         return np.zeros(config.EMBEDDING_DIM)
#
#     # 对每个块进行编码
#     chunk_embeddings = []
#     for chunk in chunks:
#         # 使用两个模型分别编码
#         embedding1 = model1.encode(
#             chunk,
#             convert_to_numpy=True,
#             normalize_embeddings=True,  # 启用模型内部归一化
#             show_progress_bar=False
#         )
#         embedding2 = model2.encode(
#             chunk,
#             convert_to_numpy=True,
#             normalize_embeddings=True,  # 启用模型内部归一化
#             show_progress_bar=False
#         )
#
#         # 拼接两个模型的结果
#         combined = np.concatenate([embedding1, embedding2])
#         chunk_embeddings.append(combined)
#
#     # 聚合分块编码结果
#     if len(chunk_embeddings) == 1:
#         return chunk_embeddings[0]
#
#     # 使用加权平均，更重视首尾块
#     weights = np.ones(len(chunk_embeddings))
#     if len(weights) > 2:
#         weights[0] = 1.5  # 加重开头
#         weights[-1] = 1.5  # 加重结尾
#
#     return np.average(chunk_embeddings, axis=0, weights=weights)
#
#
# # 批量编码函数
# def encode_batch(
#         batch_texts: List[str],
#         model1: SentenceTransformer,
#         model2: SentenceTransformer,
#         config: Config
# ) -> np.ndarray:
#     """批量编码长文本"""
#     batch_embeddings = []
#
#     for text in batch_texts:
#         embedding = encode_long_text(text, model1, model2, config)
#         batch_embeddings.append(embedding)
#
#     return np.array(batch_embeddings)
#
#
# # 主函数
# def main():
#     data_file_path = '/data/cclsol/wk/LLMRec-main1/data/netflix/user_preference.pkl'
#     output_file = '/data/cclsol/wk/LLMRec-main1/data/netflix/user_music_profiles_embedding_dict新电影'
#     ui_graph_path = '/data/cclsol/wk/LLMRec-main1/data/netflix/train_mat'
#     config = Config()
#
#     # 加载模型
#     print("加载模型...")
#     model1 = SentenceTransformer('./pretrain')
#     model2 = SentenceTransformer('./pretrain1')
#
#     # 设置模型最大序列长度
#     model1.max_seq_length = config.MAX_SEQ_LENGTH
#     model2.max_seq_length = config.MAX_SEQ_LENGTH
#
#     # 验证模型功能
#     test_texts = ["用户喜欢摇滚音乐和流行歌曲", "用户偏爱古典音乐和爵士乐"]
#     test_embeddings = encode_batch(test_texts, model1, model2, config)
#
#     # 计算测试文本相似度
#     similarity = np.dot(test_embeddings[0], test_embeddings[1]) / (
#             np.linalg.norm(test_embeddings[0]) * np.linalg.norm(test_embeddings[1])
#     )
#     print(f"测试文本相似度: {similarity:.4f}")
#
#     # 加载用户画像数据
#     print(f"加载用户画像数据: {data_file_path}")
#     user_profiles = pickle.load(open(data_file_path, 'rb'))
#
#     # 加载UI图数据获取目标用户ID
#     print(f"加载UI图数据: {ui_graph_path}")
#     try:
#         with open(ui_graph_path, 'rb') as f:
#             ui_graph = pickle.load(f)
#         target_user_ids = list(range(ui_graph.shape[0]))
#         print(f"目标用户数量: {len(target_user_ids)}")
#     except Exception as e:
#         print(f"无法加载UI图数据: {e}")
#         print("使用默认用户ID列表")
#         target_user_ids = list(range(1892))
#
#     # 准备编码数据
#     print("处理用户画像文本...")
#     user_texts = {}
#     for user_id, profile in tqdm(user_profiles.items(), desc="处理用户文本"):
#         # 合并用户所有属性为一个文本
#         all_texts = []
#         for attr_name, attr_value in profile.items():
#             processed_text = process_text(attr_value)
#             if processed_text:
#                 all_texts.append(f"{attr_name}: {processed_text}")
#
#         # 连接所有属性文本
#         user_text = ' '.join(all_texts)
#         user_texts[user_id] = user_text
#
#     # 编码用户画像
#     print("开始编码用户画像...")
#     encoded_data = {}
#     user_ids = list(user_texts.keys())
#     num_batches = (len(user_ids) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
#
#     for batch_idx in tqdm(range(num_batches), desc="编码批次"):
#         start_idx = batch_idx * config.BATCH_SIZE
#         end_idx = min(start_idx + config.BATCH_SIZE, len(user_ids))
#         batch_ids = user_ids[start_idx:end_idx]
#
#         # 获取批次文本
#         batch_texts = [user_texts[user_id] for user_id in batch_ids]
#
#         # 编码并归一化
#         batch_embeddings = encode_batch(batch_texts, model1, model2, config)
#         normalized_embeddings = normalize_embeddings(batch_embeddings, *config.NORMALIZE_RANGE)
#
#         # 保存结果
#         for i, user_id in enumerate(batch_ids):
#             encoded_data[user_id] = normalized_embeddings[i]
#
#     print(f"编码完成，共处理 {len(encoded_data)} 个用户")
#
#     # 验证编码结果
#     sample_ids = list(encoded_data.keys())[:5]
#     print("\n编码结果示例:")
#     for user_id in sample_ids:
#         embedding = encoded_data[user_id]
#         print(f"用户ID: {user_id}, 嵌入维度: {embedding.shape}, 范数: {np.linalg.norm(embedding):.4f}")
#
#     # 检查相似度
#     if len(sample_ids) >= 2:
#         sim = np.dot(encoded_data[sample_ids[0]], encoded_data[sample_ids[1]]) / (
#                 np.linalg.norm(encoded_data[sample_ids[0]]) * np.linalg.norm(encoded_data[sample_ids[1]])
#         )
#         print(f"前两个用户相似度: {sim:.4f}")
#
#     # 对齐用户嵌入
#     print(f"对齐用户嵌入到目标长度: {len(target_user_ids)}")
#     default_embedding = np.zeros(config.EMBEDDING_DIM, dtype=np.float64)
#     aligned_embeddings = {
#         user_id: encoded_data.get(user_id, default_embedding)
#         for user_id in target_user_ids
#     }
#
#     print(f"对齐后用户嵌入数量: {len(aligned_embeddings)}")
#
#     # 保存编码结果
#     print(f"保存编码结果到 {output_file}")
#     with open(output_file, 'wb') as f:
#         pickle.dump(aligned_embeddings, f)
#
#     print("完成!")
#
#
# if __name__ == "__main__":
#     main()