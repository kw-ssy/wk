import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--alpha_bpr', type=float, default=0.5, help='BPR')
    parser.add_argument('--seq_length', type=int, default=5, help='Length of user interaction sequence.')
    parser.add_argument('--codebook_size', type=int, default=256, help='Size of the codebook.')
    parser.add_argument('--code_dim', type=int, default=64, help='Dimension of each codebook entry.')
    parser.add_argument('--data_path', nargs='?', default='./data/', help='Input data path')  
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='ml-1m', help='Choose a dataset from {movieLens, netflix}')
    parser.add_argument('--verbose', type=int, default=5, help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epoch.')  
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='Regularizations.')  
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')                     
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')
    parser.add_argument('--early_stopping_patience', type=int, default=7, help='Early Stop Patience')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')   
    parser.add_argument('--debug', action='store_true')  
    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation') 
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')  
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--sc', type=float, default=1.0, help='GCN self connection')
    parser.add_argument('--feat_reg_decay', default=1e-5, type=float, help='Feature Reg Decay') 
    parser.add_argument('--title', default="try_to_draw_line", type=str, help='')  
    parser.add_argument('--cf_model', nargs='?', default='lightgcn', help='Downstream Collaborative Filtering model {mf, ngcf, lightgcn, mmgcn, vbpr, hafr, bm3}')   
    parser.add_argument('--point', default="", type=str, help='')  #

    # train
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')  
    parser.add_argument('--de_lr', type=float, default=0.0002, help='Decoder learning rate.')  
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight_decay')  #

    # 知识图谱嵌入模型相关参数
    parser.add_argument('--kg_embedding_dim', type=int, default=256,
                        help='Embedding dimension for knowledge graph entities and relations.')
    parser.add_argument('--kg_loss_rate', type=float, default=0.5,
                        help='Weight of the knowledge graph loss in the total loss.')
    parser.add_argument('--kg_cat_rate', type=float, default=0.2,
                        help='Rate of knowledge graph feature fusion in user and item embeddings.')
    parser.add_argument('--prune_ratio', type=float, default=0.9, help='知识图谱剪枝比例')
    parser.add_argument('--prune_interval', type=int, default=8, help='知识图谱剪枝间隔(epoch)')
    # 知识图谱评估与剪枝相关参数
    parser.add_argument('--kg_threshold', type=float, default=0.35, help='知识图谱三元组预测阈值')
    parser.add_argument('--user_prune_bonus', type=float, default=0.5, help='用户相关三元组剪枝优先级')
    parser.add_argument('--item_prune_bonus', type=float, default=0.5, help='物品相关三元组剪枝优先级')
    # # 区分用户/物品实体的超参数
    parser.add_argument('--user_kg_rate', type=float, default=0.6, help='用户实体嵌入融合权重')
    parser.add_argument('--item_kg_rate', type=float, default=0.6, help='物品实体嵌入融合权重')
    # parser.add_argument('--user_kg_loss_weight', type=float, default=0.4, help='用户相关三元组损失权重')
    # parser.add_argument('--item_kg_loss_weight', type=float, default=0.4, help='物品相关三元组损失权重')
    #交叉注意力模块
    parser.add_argument('--contrast_weight', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--cross_attn_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--tau', type=float, default=0.05, help='Temperature for contrastive loss')
    parser.add_argument('--warmup_epochs', type=int, default=8, help='Number of warmup epochs')
    #多层跨模态对齐组件参数
    parser.add_argument('--id_direct_weight', type=float, default=0.1, help='Weight for ID direct guidance')
    parser.add_argument('--id_indirect_weight', type=float, default=0.1, help='Weight for ID indirect guidance')
    parser.add_argument('--modal_direct_weight', type=float, default=0.1, help='Weight for modal direct alignment')
    parser.add_argument('--modal_indirect_weight', type=float, default=1.0, help='Weight for modal indirect alignment')
    # model
    parser.add_argument('--layers', type=int, default=1, help='Number of graph conv layers')
    parser.add_argument('--social_rate', type=float, default=0.1, help='社交特征在用户嵌入中的权重。')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--mask_rate', type=float, default=0.0, help='Mask rate')   
    parser.add_argument('--mask', type=bool, default=False, help='If mask')   
    parser.add_argument('--user_cat_rate', type=float, default=2.8, help='User cat rate')
    parser.add_argument('--item_cat_rate', type=float, default=0.005, help='Item cat rate')
    parser.add_argument('--model_cat_rate', type=float, default=0.02, help='Model cat rate')
    parser.add_argument('--de_drop1', default=0.31, type=float, help='for D model2')  #
    parser.add_argument('--de_drop2', default=0.5, type=float, help='')  #

    # loss
    parser.add_argument('--aug_mf_rate', type=float, default=0.012, help='Augmentation mf rate')      # 
    parser.add_argument('--prune_loss_drop_rate', type=float, default=0.71, help='Prune loss drop rate')
    parser.add_argument('--mm_mf_rate', type=float, default=0.0001, help='MM mf rate')    
    parser.add_argument('--feat_loss_type', default="sce", type=str, help='Feature loss type')  #
    parser.add_argument('--att_re_rate', type=float, default=0.00000, help='Attribute restoration rate')      # 
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
    parser.add_argument('--aug_sample_rate', type=float, default=0.1, help='Augmentation sample rate')
    parser.add_argument('--mf_emb_rate', type=float, default=0.0, help='MF embedding rate')

    return parser.parse_args()
