import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
from utilityml.parser import parse_args


class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train电影训练集.json'
        val_file = path + '/val电影验证集.json'
        test_file = path + '/test电影测试集.json'

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []

        # 用于存储用户 ID 和物品 ID 的映射关系
        self.user_id_mapping = {}
        self.item_id_mapping = {}

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))

        # 收集所有的用户 ID 和物品 ID
        all_user_ids = set()
        all_item_ids = set()
        for uid, items in train.items():
            uid = int(uid)
            all_user_ids.add(uid)
            all_item_ids.update(items)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            all_user_ids.add(uid)
            all_item_ids.update(items)
            self.n_test += len(items)

        for uid, items in val.items():
            uid = int(uid)
            all_user_ids.add(uid)
            all_item_ids.update(items)

        # 建立用户 ID 和物品 ID 的映射关系
        self.n_users = len(all_user_ids)
        self.n_items = len(all_item_ids)
        self.user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_user_ids)}
        self.item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_item_ids)}

        self.exist_users = list(all_user_ids)

        self.print_statistics()

        # 根据映射后的 ID 初始化矩阵
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}

        # 对训练数据中的 ID 进行映射转换并填充矩阵
        for uid, train_items in train.items():
            uid = int(uid)
            mapped_uid = self.user_id_mapping[uid]
            mapped_train_items = [self.item_id_mapping[i] for i in train_items]
            self.train_items[mapped_uid] = mapped_train_items
            for i in mapped_train_items:
                self.R[mapped_uid, i] = 1.

        # 对测试数据中的 ID 进行映射转换
        for uid, test_items in test.items():
            uid = int(uid)
            mapped_uid = self.user_id_mapping[uid]
            mapped_test_items = [self.item_id_mapping[i] for i in test_items]
            self.test_set[mapped_uid] = mapped_test_items

        # 对验证数据中的 ID 进行映射转换
        for uid, val_items in val.items():
            uid = int(uid)
            mapped_uid = self.user_id_mapping[uid]
            mapped_val_items = [self.item_id_mapping[i] for i in val_items]
            self.val_set[mapped_uid] = mapped_val_items

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat电影.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat电影.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat电影.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat电影.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat电影.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat电影.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def sample(self):
        # 使用映射后的用户ID集合
        valid_users = list(self.train_items.keys())

        # 确保批大小不超过有效用户数
        if self.batch_size > len(valid_users):
            print(f"警告: 批大小({self.batch_size})大于有效用户数({len(valid_users)})，将调整为有效用户数")
            self.batch_size = len(valid_users)

        # 从有效用户中采样
        users = rd.sample(valid_users, self.batch_size)

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]  # 现在u是映射后的用户ID，不会报错
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))