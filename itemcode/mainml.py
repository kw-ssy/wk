from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch import autograd
import random
from sklearn.model_selection import train_test_split
import copy
# from itertools import product  # 注释原网格搜索导入
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utilityml.parser import parse_args
from Modelsml import MM_Model, Decoder
from utilityml.batch_test import *
from utilityml.logging import Logger
from utilityml.norm import build_sim, build_knn_normalized_graph

import setproctitle

args = parse_args()


# 加载知识图谱数据并处理
# 加载知识图谱数据并处理（增加用户/物品实体区分）
def load_kg_data(file_path):
    kg_triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                head, relation, tail = line.strip().split('\t')
                kg_triples.append((head, relation, tail))
            except ValueError:
                print(f"跳过不符合格式的行: {line}")
    return kg_triples


kg_file_path = '/data/cclsol/wk/LLMRec-main1/data/ml-1m/combined_knowledge_graph电影.txt'
kg_triples = load_kg_data(kg_file_path)

# 构建实体和关系的映射（区分用户/物品实体）
entities = set()
relations = set()
USER_PREFIX = "User_"  # 用户实体前缀
ITEM_PREFIX = "Movie_"  # 物品实体前缀

# 分离用户实体、物品实体和其他实体
user_entities = set()
item_entities = set()
other_entities = set()

for head, relation, tail in kg_triples:
    entities.add(head)
    entities.add(tail)
    relations.add(relation)

    # 标记用户实体（含User_前缀）
    if head.startswith(USER_PREFIX):
        user_entities.add(head)
    if tail.startswith(USER_PREFIX):
        user_entities.add(tail)
    # 标记物品实体（含Item_前缀）
    if head.startswith(ITEM_PREFIX):
        item_entities.add(head)
    if tail.startswith(ITEM_PREFIX):
        item_entities.add(tail)

# 实体ID映射（保留用户/物品实体的单独映射）
entity2id = {entity: idx for idx, entity in enumerate(entities)}
relation2id = {relation: idx for idx, relation in enumerate(relations)}

# 用户/物品实体ID列表（用于后续模型融合）
user_entity_ids = [entity2id[ent] for ent in user_entities]
item_entity_ids = [entity2id[ent] for ent in item_entities]
print(
    f"区分后实体数量 - 用户: {len(user_entity_ids)}, 物品: {len(item_entity_ids)}, 其他: {len(entities) - len(user_entity_ids) - len(item_entity_ids)}")

# 三元组转换为ID形式，并区分用户/物品相关三元组
kg_triples_id = []
user_related_triples = []  # 用户属性/关系三元组（如User_1-age-25）
item_related_triples = []  # 物品属性/关系三元组（如Item_1-genre-Action）
for head, relation, tail in kg_triples:
    head_id = entity2id[head]
    relation_id = relation2id[relation]
    tail_id = entity2id[tail]
    kg_triples_id.append((head_id, relation_id, tail_id))

    # 标记用户相关三元组（头或尾为用户实体）
    if head in user_entities or tail in user_entities:
        user_related_triples.append((head_id, relation_id, tail_id))
    # 标记物品相关三元组（头或尾为物品实体）
    if head in item_entities or tail in item_entities:
        item_related_triples.append((head_id, relation_id, tail_id))

# 划分训练集和测试集（区分用户/物品三元组）
train_kg, test_kg = train_test_split(kg_triples_id, test_size=0.2, random_state=42)
train_user_kg, test_user_kg = train_test_split(user_related_triples, test_size=0.2, random_state=42)
train_item_kg, test_item_kg = train_test_split(item_related_triples, test_size=0.2, random_state=42)


# 知识图谱嵌入模型（保持不变，增加用户/物品实体区分的注释）
class TransE(nn.Module):
    def __init__(self, entity_num, relation_num, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(entity_num, embedding_dim)  # 包含用户/物品/其他实体
        self.relation_embeddings = nn.Embedding(relation_num, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, heads, relations, tails):
        head_embeds = self.entity_embeddings(heads)
        relation_embeds = self.relation_embeddings(relations)
        tail_embeds = self.entity_embeddings(tails)
        score = torch.norm(head_embeds + relation_embeds - tail_embeds, p=2, dim=1)
        return score


entity_num = len(entities)
relation_num = len(relations)
embedding_dim = 128
transe_model = TransE(entity_num, relation_num, embedding_dim).cuda()


class Trainer(object):
    def __init__(self, data_config):

        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.train_user_kg = train_user_kg
        self.train_item_kg = train_item_kg
        self.test_user_kg = test_user_kg
        self.test_item_kg = test_item_kg

        self.image_feats = np.load(args.data_path + '{}/image_feat电影.npy'.format(args.dataset))
        self.text_feats = np.load(args.data_path + '{}/text_feat_movies.npy'.format(args.dataset))
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        self.ui_graph = self.ui_graph_raw = pickle.load(open(args.data_path + args.dataset + '/train_mat电影', 'rb'))
        # get user embedding
        augmented_user_init_embedding = pickle.load(open(args.data_path + args.dataset + '/movie_user_profiles新电影', 'rb'))
        augmented_user_init_embedding_list = []
        for user_id in augmented_user_init_embedding.keys():
            # 获取该用户的嵌入向量
            embedding = augmented_user_init_embedding[user_id]
            augmented_user_init_embedding_list.append(embedding)

        augmented_user_init_embedding_final = np.array(augmented_user_init_embedding_list)
        pickle.dump(augmented_user_init_embedding_final, open(args.data_path + args.dataset + '/user_profiles_init_embedding_final', 'wb'))
        self.user_init_embedding = pickle.load(open(args.data_path + args.dataset + '/user_profiles_init_embedding_final', 'rb'))
        # get separate embedding matrix
        if args.dataset == 'ml-1m1':
            augmented_total_embed_dict = {'title': [], 'director': [], 'country': [], 'language': [],
                                          'starring': [], 'popularity': []}
        elif args.dataset == 'ml-1m':
            augmented_total_embed_dict = {'title': [], 'summarization': [], 'reasoning': []}
        elif args.dataset == 'netflix':
            augmented_total_embed_dict = {'year': [], 'title': [], 'director': [], 'country': [], 'language': [],
                                          'starring': [], 'popularity': []}
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        required_keys = ['title', 'summarization', 'reasoning']
        for key in required_keys:
            if key not in augmented_total_embed_dict:
                augmented_total_embed_dict[key] = []

        augmented_atttribute_embedding_dict = pickle.load(open(args.data_path + args.dataset + '/augmented_attribute_embedding_dict新电影', 'rb'))
        for value in augmented_atttribute_embedding_dict.keys():
            for i in range(len(augmented_atttribute_embedding_dict[value])):
                augmented_total_embed_dict[value].append(augmented_atttribute_embedding_dict[value][i])
            augmented_total_embed_dict[value] = np.array(augmented_total_embed_dict[value])
        pickle.dump(augmented_total_embed_dict, open(args.data_path + args.dataset + '/augmented_total_embed_dict', 'wb'))
        self.item_attribute_embedding = pickle.load(open(args.data_path + args.dataset + '/augmented_total_embed_dict', 'rb'))

        self.image_ui_index = {'x': [], 'y': []}
        self.text_ui_index = {'x': [], 'y': []}

        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.iu_graph = self.ui_graph.T

        self.ui_graph = self.csr_norm(self.ui_graph, mean_flag=True)
        self.iu_graph = self.csr_norm(self.iu_graph, mean_flag=True)
        self.ui_graph = self.matrix_to_tensor(self.ui_graph)
        self.user_entity_ids = user_entity_ids
        self.item_entity_ids = item_entity_ids
        self.iu_graph = self.matrix_to_tensor(self.iu_graph)
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        self.model_mm = MM_Model(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout,
                                 self.image_feats, self.text_feats, self.user_init_embedding,
                                 self.item_attribute_embedding, entity_num, entity2id)
        self.model_mm = self.model_mm.cuda()
        self.decoder = Decoder(self.user_init_embedding.shape[1]).cuda()

        self.optimizer = optim.AdamW(
            [
                {'params': self.model_mm.parameters()},
                {'params': transe_model.parameters()}
            ], lr=self.lr)

        self.de_optimizer = optim.AdamW(
            [
                {'params': self.decoder.parameters()},
            ], lr=args.de_lr)

        # 添加剪枝参数
        self.prune_ratio = args.prune_ratio  # 剪枝比例（保留前prune_ratio的三元组）
        self.prune_interval = args.prune_interval  # 剪枝间隔（epoch数）
        self.current_epoch = 0

        # 初始化剪枝后的知识图谱
        self.pruned_kg_triples_id = kg_triples_id.copy()
        self.train_kg_triples_id, self.test_kg_triples_id = train_test_split(
            self.pruned_kg_triples_id, test_size=0.2, random_state=42
        )

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)
        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)
        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #

    def innerProduct(self, u_pos, i_pos, u_neg, j_neg):
        pred_i = torch.sum(torch.mul(u_pos, i_pos), dim=-1)
        pred_j = torch.sum(torch.mul(u_neg, j_neg), dim=-1)
        return pred_i, pred_j

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        feat_reg = 1. / 2 * (g_item_image ** 2).sum() + 1. / 2 * (g_item_text ** 2).sum() \
                   + 1. / 2 * (g_user_image ** 2).sum() + 1. / 2 * (g_user_text ** 2).sum()
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = args.feat_reg_decay * feat_reg
        return feat_emb_loss

    def prune_loss(self, pred, drop_rate):
        ind_sorted = np.argsort(pred.cpu().data).cuda()
        loss_sorted = pred[ind_sorted]
        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        loss_update = pred[ind_update]
        return loss_update.mean()

    def mse_criterion(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        tmp_loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        tmp_loss = tmp_loss.mean()
        loss = F.mse_loss(x, y)
        return loss

    def sce_criterion(self, x, y, alpha=1):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def test(self, users_to_test, is_val):
        self.model_mm.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings, *rest = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph,
                                                                self.image_iu_graph, self.text_ui_graph,
                                                                self.text_iu_graph)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):

        now_time = datetime.now()
        run_time = datetime.strftime(now_time, '%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        stopping_step = 0

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            contrastive_loss = 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.
            build_item_graph = True

            self.gene_u, self.gene_real, self.gene_fake = None, None, {}
            self.topk_p_dict, self.topk_id_dict = {}, {}

            for idx in tqdm(range(n_batch)):
                self.model_mm.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()

                # augment samples
                augmented_sample_dict = pickle.load(
                    open(args.data_path + args.dataset + '/augmented_user_movie_samples', 'rb')
                )

                # 从增强样本字典的键创建用户ID到索引的映射
                # 假设augmented_sample_dict的键是原始用户ID（如71363）
                # 创建一个映射表：原始ID -> 连续索引（0,1,2...）
                user_id_to_index = {user_id: idx for idx, user_id in enumerate(augmented_sample_dict.keys())}

                # 从用户列表中采样（注意users应该是原始ID列表）
                users_aug = random.sample(users, int(len(users) * args.aug_sample_rate))

                # 初始化有效样本列表
                valid_users_aug = []
                pos_items_aug = []
                neg_items_aug = []

                # 处理每个采样的用户
                for user in users_aug:
                    # 检查用户是否在增强样本字典中
                    if user in augmented_sample_dict:
                        # 获取增强样本
                        item0, item1 = augmented_sample_dict[user]

                        # 检查物品索引是否有效
                        if item0 < self.n_items and item1 < self.n_items:
                            valid_users_aug.append(user)  # 保留原始ID用于后续处理
                            pos_items_aug.append(item0)
                            neg_items_aug.append(item1)

                # 更新批大小和样本列表
                self.new_batch_size = len(valid_users_aug)
                users += valid_users_aug  # 使用原始ID
                pos_items += pos_items_aug
                neg_items += neg_items_aug

                sample_time += time() - sample_t1
                user_presentation_h, item_presentation_h, image_i_feat, text_i_feat, image_u_feat, text_u_feat \
                    , user_prof_feat_pre, item_prof_feat_pre, user_prof_feat, item_prof_feat, user_att_feats, item_att_feats, i_mask_nodes, u_mask_nodes \
                    = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph,
                                    self.text_ui_graph, self.text_iu_graph)

                u_bpr_emb = user_presentation_h[users]
                i_bpr_pos_emb = item_presentation_h[pos_items]
                i_bpr_neg_emb = item_presentation_h[neg_items]
                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_bpr_emb, i_bpr_pos_emb, i_bpr_neg_emb)

                # modal feat
                image_u_bpr_emb = image_u_feat[users]
                image_i_bpr_pos_emb = image_i_feat[pos_items]
                image_i_bpr_neg_emb = image_i_feat[neg_items]
                image_batch_mf_loss, image_batch_emb_loss, image_batch_reg_loss = self.bpr_loss(image_u_bpr_emb,
                                                                                                image_i_bpr_pos_emb,
                                                                                                image_i_bpr_neg_emb)
                text_u_bpr_emb = text_u_feat[users]
                text_i_bpr_pos_emb = text_i_feat[pos_items]
                text_i_bpr_neg_emb = text_i_feat[neg_items]
                text_batch_mf_loss, text_batch_emb_loss, text_batch_reg_loss = self.bpr_loss(text_u_bpr_emb,
                                                                                              text_i_bpr_pos_emb,
                                                                                              text_i_bpr_neg_emb)
                mm_mf_loss = image_batch_mf_loss + text_batch_mf_loss

                batch_mf_loss_aug = 0
                for index, value in enumerate(item_att_feats):  #
                    u_g_embeddings_aug = user_prof_feat[users]
                    pos_i_g_embeddings_aug = item_att_feats[value][pos_items]
                    neg_i_g_embeddings_aug = item_att_feats[value][neg_items]
                    tmp_batch_mf_loss_aug, batch_emb_loss_aug, batch_reg_loss_aug = self.bpr_loss(u_g_embeddings_aug,
                                                                                                   pos_i_g_embeddings_aug,
                                                                                                   neg_i_g_embeddings_aug)
                    batch_mf_loss_aug += tmp_batch_mf_loss_aug

                feat_emb_loss = self.feat_reg_loss_calculation(image_i_feat, text_i_feat, image_u_feat, text_u_feat)

                att_re_loss = 0
                if args.mask:
                    input_i = {}
                    for index, value in enumerate(item_att_feats.keys()):
                        input_i[value] = item_att_feats[value][i_mask_nodes]
                    decoded_u, decoded_i = self.decoder(torch.tensor(user_prof_feat[u_mask_nodes]), input_i)
                    if args.feat_loss_type == 'mse':
                        att_re_loss += self.mse_criterion(decoded_u, torch.tensor(self.user_init_embedding[u_mask_nodes]).cuda(),
                                                          alpha=args.alpha_l)
                        for index, value in enumerate(item_att_feats.keys()):
                            att_re_loss += self.mse_criterion(decoded_i[index],
                                                              torch.tensor(self.item_attribute_embedding[value][i_mask_nodes]).cuda(),
                                                              alpha=args.alpha_l)
                    elif args.feat_loss_type == 'sce':
                        att_re_loss += self.sce_criterion(decoded_u, torch.tensor(self.user_init_embedding[u_mask_nodes]).cuda(),
                                                          alpha=args.alpha_l)
                        for index, value in enumerate(item_att_feats.keys()):
                            att_re_loss += self.sce_criterion(decoded_i[index],
                                                              torch.tensor(self.item_attribute_embedding[value][i_mask_nodes]).cuda(),
                                                              alpha=args.alpha_l)

                if len(self.train_kg_triples_id) < self.batch_size:
                    sample_size = len(self.train_kg_triples_id)
                else:
                    sample_size = self.batch_size
                # 知识图谱损失计算
                kg_loss = 0.0
                sample_size = min(self.batch_size, len(train_kg))
                if sample_size > 0:
                    # 1. 整体三元组损失
                    heads, relations, tails = zip(*random.sample(train_kg, sample_size))
                    heads = torch.tensor(heads).cuda()
                    relations = torch.tensor(relations).cuda()
                    tails = torch.tensor(tails).cuda()
                    total_kg_loss = transe_model(heads, relations, tails).mean()

                    # 2. 用户相关三元组损失（权重更高）
                    user_sample_size = min(int(sample_size * 0.4), len(self.train_user_kg))
                    if user_sample_size > 0:
                        u_heads, u_rels, u_tails = zip(*random.sample(self.train_user_kg, user_sample_size))
                        u_heads = torch.tensor(u_heads).cuda()
                        u_rels = torch.tensor(u_rels).cuda()
                        u_tails = torch.tensor(u_tails).cuda()
                        user_kg_loss = transe_model(u_heads, u_rels, u_tails).mean()
                    else:
                        user_kg_loss = 0.0

                    # 3. 物品相关三元组损失（权重更高）
                    item_sample_size = min(int(sample_size * 0.4), len(self.train_item_kg))
                    if item_sample_size > 0:
                        i_heads, i_rels, i_tails = zip(*random.sample(self.train_item_kg, item_sample_size))
                        i_heads = torch.tensor(i_heads).cuda()
                        i_rels = torch.tensor(i_rels).cuda()
                        i_tails = torch.tensor(i_tails).cuda()
                        item_kg_loss = transe_model(i_heads, i_rels, i_tails).mean()
                    else:
                        item_kg_loss = 0.0

                    # 总知识图谱损失（用户/物品三元组权重更高）
                    kg_loss = 0.2 * total_kg_loss + 0.4 * user_kg_loss + 0.4 * item_kg_loss

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + feat_emb_loss + args.aug_mf_rate * batch_mf_loss_aug + args.mm_mf_rate * mm_mf_loss + args.att_re_rate * att_re_loss + args.kg_loss_rate * kg_loss
                nn.utils.clip_grad_norm_(self.model_mm.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(transe_model.parameters(), max_norm=1.0)
                self.optimizer.zero_grad()
                batch_loss.backward(retain_graph=False)

                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

            del user_presentation_h, item_presentation_h, u_bpr_emb, i_bpr_neg_emb, i_bpr_pos_emb

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, contrastive_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_test, is_val=False)  # ^-^
            training_time_list.append(t2 - t1)

            t3 = time()

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                           'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][1],
                            ret['recall'][2],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][-1],
                            ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][-1])
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (
                    eval(args.Ks)[1], test_ret['recall'][1], test_ret['precision'][1], test_ret['ndcg'][1]))
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break

            # 定期进行知识图谱剪枝
            if (epoch + 1) % self.prune_interval == 0:
                self.prune_kg()

        # 评估知识图谱利用效果
        self.evaluate_kg_utilization(test_kg)

        self.logger.logging(str(test_ret))

        return best_recall

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / (2 * (users ** 2).sum() + 1e-8) + 1. / (2 * (pos_items ** 2).sum() + 1e-8) + 1. / (
                2 * (neg_items ** 2).sum() + 1e-8)
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores + 1e-8)
        mf_loss = - self.prune_loss(maxi, args.prune_loss_drop_rate)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def evaluate_kg_utilization(self, test_kg_triples):
        """
        评估知识图谱的利用效果，包括特定关系预测准确性、长尾实体与关系处理能力、知识图谱信息覆盖率
        区分用户实体和物品实体的评估结果
        :param test_kg_triples: 测试用的知识图谱三元组
        """
        self.logger.logging(f"KG Size: {len(self.pruned_kg_triples_id)}/{len(kg_triples_id)}")

        # 1. 特定关系预测准确性（区分用户关系和物品关系）
        user_relations = [relation2id.get('user_likes', 0), relation2id.get('user_age', 0)]  # 示例用户关系
        item_relations = [relation2id.get('item_genre', 0), relation2id.get('item_director', 0)]  # 示例物品关系

        user_rel_acc = self.specific_relation_accuracy(user_relations, test_kg_triples)
        item_rel_acc = self.specific_relation_accuracy(item_relations, test_kg_triples)
        self.logger.logging(f"User Relation Accuracy: {user_rel_acc:.4f}, Item Relation Accuracy: {item_rel_acc:.4f}")

        # 2. 长尾实体与关系处理能力（区分用户/物品长尾）
        long_tail_user = self.long_tail_performance_improvement(test_kg_triples, entity_type='user')
        long_tail_item = self.long_tail_performance_improvement(test_kg_triples, entity_type='item')
        self.logger.logging(
            f"Long Tail User Performance: {long_tail_user:.4f}, Long Tail Item Performance: {long_tail_item:.4f}")

        # 3. 知识图谱信息覆盖率（区分用户/物品实体）
        user_coverage, user_rel_coverage = self.kg_information_coverage(test_kg_triples, entity_type='user')
        item_coverage, item_rel_coverage = self.kg_information_coverage(test_kg_triples, entity_type='item')
        self.logger.logging(
            f"User Entity Coverage: {user_coverage:.4f}, User Relation Coverage: {user_rel_coverage:.4f}")
        self.logger.logging(
            f"Item Entity Coverage: {item_coverage:.4f}, Item Relation Coverage: {item_rel_coverage:.4f}")

    def specific_relation_accuracy(self, specific_relations, test_kg_triples):
        """
        计算特定关系的预测准确性
        :param specific_relations: 要评估的特定关系列表
        :param test_kg_triples: 测试用的知识图谱三元组
        :return: 特定关系的预测准确率
        """
        if not specific_relations:
            return 0.0

        correct_count = 0
        total_count = 0

        # 筛选包含特定关系的三元组
        kg_subset = [triple for triple in test_kg_triples if triple[1] in specific_relations]

        # 限制评估样本量
        if len(kg_subset) > 10000:
            kg_subset = random.sample(kg_subset, 10000)

        for head, relation, tail in kg_subset:
            heads = torch.tensor([head]).cuda()
            relations = torch.tensor([relation]).cuda()
            tails = torch.tensor([tail]).cuda()

            # 计算三元组得分（越小越好）
            score = transe_model(heads, relations, tails)

            # 使用更严格的阈值判断（可通过args调整）
            prediction = score < args.kg_threshold  # 默认0.5
            if prediction:
                correct_count += 1
            total_count += 1

        if total_count == 0:
            return 0.0
        return correct_count / total_count

    def long_tail_performance_improvement(self, test_kg_triples, entity_type='all'):
        """
        评估模型对长尾实体和关系的处理能力
        :param test_kg_triples: 测试用的知识图谱三元组
        :param entity_type: 'all'/'user'/'item'，指定评估的实体类型
        :return: 长尾实体和关系的处理准确率
        """
        # 筛选特定类型的实体
        if entity_type == 'user':
            target_entities = set(self.user_entity_ids)
        elif entity_type == 'item':
            target_entities = set(self.item_entity_ids)
        else:
            target_entities = set(range(entity_num))  # 所有实体

        # 统计实体和关系频率
        entity_freq = defaultdict(int)
        relation_freq = defaultdict(int)

        for head, rel, tail in test_kg_triples:
            if head in target_entities:
                entity_freq[head] += 1
            if tail in target_entities:
                entity_freq[tail] += 1
            relation_freq[rel] += 1

        # 计算长尾阈值（后20%作为长尾）
        entity_counts = np.array(list(entity_freq.values()))
        if len(entity_counts) == 0:
            return 0.0

        entity_threshold = np.quantile(entity_counts, 0.2)

        # 构建长尾实体列表
        long_tail_entities = [e for e, c in entity_freq.items() if c < entity_threshold]

        # 筛选包含长尾实体的三元组
        kg_test = [triple for triple in test_kg_triples
                   if (triple[0] in long_tail_entities or triple[2] in long_tail_entities)]

        # 限制评估样本量
        if len(kg_test) > 5000:
            kg_test = random.sample(kg_test, 5000)

        correct_count = 0
        for head, rel, tail in kg_test:
            heads = torch.tensor([head]).cuda()
            relations = torch.tensor([rel]).cuda()
            tails = torch.tensor([tail]).cuda()

            score = transe_model(heads, relations, tails)
            if score < args.kg_threshold:  # 使用与评估相同的阈值
                correct_count += 1

        if len(kg_test) == 0:
            return 0.0
        return correct_count / len(kg_test)

    def kg_information_coverage(self, test_kg_triples, entity_type='all', threshold=0.5):
        """
        计算知识图谱信息的覆盖率
        :param test_kg_triples: 测试用的知识图谱三元组
        :param entity_type: 'all'/'user'/'item'，指定评估的实体类型
        :param threshold: 预测阈值
        :return: 实体覆盖率和关系覆盖率
        """
        # 筛选特定类型的实体
        if entity_type == 'user':
            target_entities = set(self.user_entity_ids)
        elif entity_type == 'item':
            target_entities = set(self.item_entity_ids)
        else:
            target_entities = set(range(entity_num))  # 所有实体

        # 限制评估样本量
        if len(test_kg_triples) > 10000:
            test_kg_triples = random.sample(test_kg_triples, 10000)

        used_entities = set()
        used_relations = set()

        for head, relation, tail in test_kg_triples:
            # 仅考虑目标类型的实体
            if head not in target_entities and tail not in target_entities:
                continue

            heads = torch.tensor([head]).cuda()
            relations = torch.tensor([relation]).cuda()
            tails = torch.tensor([tail]).cuda()

            score = transe_model(heads, relations, tails)
            prediction = score < threshold

            if prediction:
                if head in target_entities:
                    used_entities.add(head)
                if tail in target_entities:
                    used_entities.add(tail)
                used_relations.add(relation)

        # 计算所有相关实体和关系
        all_entities = set()
        all_relations = set()

        for head, relation, tail in test_kg_triples:
            if head in target_entities:
                all_entities.add(head)
            if tail in target_entities:
                all_entities.add(tail)
            all_relations.add(relation)

        entity_coverage = len(used_entities) / len(all_entities) if len(all_entities) > 0 else 0
        relation_coverage = len(used_relations) / len(all_relations) if len(all_relations) > 0 else 0

        return entity_coverage, relation_coverage

    def prune_kg(self):
        """基于当前模型对知识图谱进行剪枝排序，优先保留用户/物品相关三元组"""
        self.model_mm.eval()  # 切换为评估模式

        # 分批处理参数
        batch_size = 1024  # 根据GPU内存调整
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 存储所有三元组的得分和BPR损失
        all_scores = []
        all_neg_scores = []

        # 记录每个三元组是否是用户/物品相关
        is_user_related = []
        is_item_related = []

        with torch.no_grad():
            # 获取三元组总数
            num_triples = len(self.pruned_kg_triples_id)

            # 分批处理三元组
            for i in range(0, num_triples, batch_size):
                # 获取当前批次的三元组
                batch_triples = self.pruned_kg_triples_id[i:i + batch_size]

                # 标记用户/物品相关三元组
                batch_user_related = [1 if (t[0] in self.user_entity_ids or t[2] in self.user_entity_ids) else 0
                                      for t in batch_triples]
                batch_item_related = [1 if (t[0] in self.item_entity_ids or t[2] in self.item_entity_ids) else 0
                                      for t in batch_triples]

                is_user_related.extend(batch_user_related)
                is_item_related.extend(batch_item_related)

                # 转换为张量并移到GPU
                batch_heads = torch.tensor([triple[0] for triple in batch_triples]).to(device)
                batch_relations = torch.tensor([triple[1] for triple in batch_triples]).to(device)
                batch_tails = torch.tensor([triple[2] for triple in batch_triples]).to(device)

                # 计算当前批次的得分
                batch_scores = transe_model(batch_heads, batch_relations, batch_tails)
                all_scores.append(batch_scores.cpu())

                # 生成负样本（随机替换头实体）
                batch_neg_heads = torch.randint(0, entity_num, batch_heads.shape).to(device)
                batch_neg_scores = transe_model(batch_neg_heads, batch_relations, batch_tails)
                all_neg_scores.append(batch_neg_scores.cpu())

                # 释放当前批次的GPU内存
                del batch_heads, batch_relations, batch_tails, batch_scores, batch_neg_heads, batch_neg_scores
                torch.cuda.empty_cache()

            # 合并所有批次的得分
            scores = torch.cat(all_scores, dim=0)
            neg_scores = torch.cat(all_neg_scores, dim=0)

            # 计算BPR损失
            bpr_loss = F.softplus(scores - neg_scores).numpy()

            # 转换为numpy数组
            is_user_related = np.array(is_user_related)
            is_item_related = np.array(is_item_related)

            # 调整损失：用户/物品相关三元组获得更高优先级（损失降低）
            adjusted_loss = bpr_loss * (
                        1 - args.user_prune_bonus * is_user_related - args.item_prune_bonus * is_item_related)

            # 根据调整后的损失进行排序（损失越小可信度越高）
            sorted_indices = np.argsort(adjusted_loss)
            keep_num = int(len(sorted_indices) * self.prune_ratio)
            kept_indices = sorted_indices[:keep_num]

            # 更新剪枝后的知识图谱
            self.pruned_kg_triples_id = [self.pruned_kg_triples_id[i] for i in kept_indices]

            # 重新划分训练集和测试集
            self.train_kg_triples_id, self.test_kg_triples_id = train_test_split(
                self.pruned_kg_triples_id, test_size=0.2, random_state=42
            )

            # 统计剪枝后的用户/物品三元组比例
            user_triples_count = sum(1 for t in self.pruned_kg_triples_id
                                     if t[0] in self.user_entity_ids or t[2] in self.user_entity_ids)
            item_triples_count = sum(1 for t in self.pruned_kg_triples_id
                                     if t[0] in self.item_entity_ids or t[2] in self.item_entity_ids)

            self.logger.logging(f"Pruned KG: Total={len(self.pruned_kg_triples_id)}, "
                                f"User Triples={user_triples_count}, Item Triples={item_triples_count}")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)

    # 定义参数空间
    # prune_ratio_space = [0.6, 0.7, 0.8, 0.9]
    # prune_interval_space = [2, 3, 4, 5]
    # kg_embedding_dim_space = [64, 128, 256]
    # kg_loss_rate_space = [0.1, 0.2, 0.3]
    # kg_cat_rate_space = [0.1, 0.15, 0.2]
    #
    # num_trials = 100 # 随机试验的次数
    # best_recall = -1
    # best_params = {}
    #
    # # 记录已尝试的参数组合
    # tried_params = set()
    #
    # for _ in range(num_trials):
    #     # 随机选择参数组合
    #     while True:
    #         prune_ratio = random.choice(prune_ratio_space)
    #         prune_interval = random.choice(prune_interval_space)
    #         kg_embedding_dim = random.choice(kg_embedding_dim_space)
    #         kg_loss_rate = random.choice(kg_loss_rate_space)
    #         kg_cat_rate = random.choice(kg_cat_rate_space)
    #         param_tuple = (prune_ratio, prune_interval, kg_embedding_dim, kg_loss_rate, kg_cat_rate)
    #         if param_tuple not in tried_params:
    #             tried_params.add(param_tuple)
    #             break
    #
    #     # 更新参数
    #     args.prune_ratio = prune_ratio
    #     args.prune_interval = prune_interval
    #     args.kg_embedding_dim = kg_embedding_dim
    #     args.kg_loss_rate = kg_loss_rate
    #     args.kg_cat_rate = kg_cat_rate

        # 初始化配置
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

        # 初始化训练器
    trainer = Trainer(data_config=config)
    trainer.train()
        # 执行训练并获取结果
        # current_recall = trainer.train()

        # 记录最佳结果
    #     if current_recall > best_recall:
    #         best_recall = current_recall
    #         best_params = {
    #             'prune_ratio': prune_ratio,
    #             'prune_interval': prune_interval,
    #             'kg_embedding_dim': kg_embedding_dim,
    #             'kg_loss_rate': kg_loss_rate,
    #             'kg_cat_rate': kg_cat_rate
    #         }
    #
    # print(f"Best Recall: {best_recall:.4f}")
    # print("Best Parameters:")
    # for key, value in best_params.items():
    #     print(f"  {key}: {value}")