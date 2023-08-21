import os
import ujson as json
import zipfile
import numpy as np
import pickle
import torch
import scipy.sparse as sp
import torch.nn as nn


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def concat_json_list(filepaths, save_path):
    json_lists = []
    for p in filepaths:
        json_lists += load_json(p)
    save_json(json_lists, save_path)


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def convert_to_seconds(hms_time):
    """ convert '00:01:12' to 72 seconds.
    :hms_time (str): time in comma separated string, e.g. '00:01:12'
    :return (int): time in seconds, e.g. 72
    """
    times = [float(t) for t in hms_time.split(":")]
    return times[0] * 3600 + times[1] * 60 + times[2]


def get_video_name_from_url(url):
    return url.split("/")[-1][:-4]


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def make_zipfile(src_dir, save_path, enclosing_dir="", exclude_dirs=None, exclude_extensions=None,
                 exclude_dirs_substring=None):
    """make a zip file of root_dir, save it to save_path.
    exclude_paths will be excluded if it is a subdir of root_dir.
    An enclosing_dir is added is specified.
    """
    abs_src = os.path.abspath(src_dir)
    with zipfile.ZipFile(save_path, "w") as zf:
        for dirname, subdirs, files in os.walk(src_dir):
            if exclude_dirs is not None:
                for e_p in exclude_dirs:
                    if e_p in subdirs:
                        subdirs.remove(e_p)
            if exclude_dirs_substring is not None:
                to_rm = []
                for d in subdirs:
                    if exclude_dirs_substring in d:
                        to_rm.append(d)
                for e in to_rm:
                    subdirs.remove(e)
            arcname = os.path.join(enclosing_dir, dirname[len(abs_src) + 1:])
            zf.write(dirname, arcname)
            for filename in files:
                if exclude_extensions is not None:
                    if os.path.splitext(filename)[1] in exclude_extensions:
                        continue  # do not zip it
                absname = os.path.join(dirname, filename)
                arcname = os.path.join(enclosing_dir, absname[len(abs_src) + 1:])
                zf.write(absname, arcname)


class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dissect_by_lengths(np_array, lengths, dim=0, assert_equal=True):
    """Dissect an array (N, D) into a list a sub-array,
    np_array.shape[0] == sum(lengths), Output is a list of nd arrays, singlton dimention is kept"""
    if assert_equal:
        assert len(np_array) == sum(lengths)
    length_indices = [0, ]
    for i in range(len(lengths)):
        length_indices.append(length_indices[i] + lengths[i])
    if dim == 0:
        array_list = [np_array[length_indices[i]:length_indices[i+1]] for i in range(len(lengths))]
    elif dim == 1:
        array_list = [np_array[:, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    elif dim == 2:
        array_list = [np_array[:, :, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    else:
        raise NotImplementedError
    return array_list


def get_ratio_from_counter(counter_obj, threshold=200):
    keys = counter_obj.keys()
    values = counter_obj.values()
    filtered_values = [counter_obj[k] for k in keys if k > threshold]
    return float(sum(filtered_values)) / sum(values)


def get_rounded_percentage(float_number, n_floats=2):
    return round(float_number * 100, n_floats)



def get_edge_frompairs(group_pairs_list, span_num, type_label=None):
    # out_matrix = torch.zeros(len(group_pairs_list), span_num, span_num).cuda()
    out_matrix = np.zeros([len(group_pairs_list), span_num, span_num])
    tril_matrix = []
    triu_matrix = []
    for idx, pairs in enumerate(group_pairs_list):
        # tril_list = []
        # triu_list = []
        for n, pair in enumerate(pairs):
            parent, child = pair
            if type_label != None:
                out_matrix[idx, child, parent] = type_label[idx][n]
            else:
                out_matrix[idx, child, parent] = 1
    
        # for i in range(span_num-1):
        # 	for j in range(i+1, span_num):
        # 		triu_list.append(out_matrix[idx, i, j])
        # triu_matrix.append(torch.cat(triu_list))
        #
        # for i in range(1, span_num):
        # 	for j in range(0, i):
        # 		tril_list.append(out_matrix[idx, i, j])
        # tril_matrix.append(torch.cat(tril_list))
    
        scr_idx, tar_idx = np.triu_indices(span_num, k=1)
        triu_matrix.append(out_matrix[idx, scr_idx, tar_idx])
        # tril_idx = np.tril_indices(span_num, k=-1)
        tril_matrix.append(out_matrix[idx, tar_idx, scr_idx])

    out_matrix = torch.Tensor(out_matrix).cuda() # [group_size, span_num, span_num]
    triu_label = torch.Tensor(triu_matrix).cuda().view(-1)  # [group_size * direct_edge]
    tril_label = torch.Tensor(tril_matrix).cuda().view(-1)  # [group_size * direct_edge]

    return out_matrix, triu_label, tril_label


def get_eval_result(res_dict):
    ARI_msg = 'ARI-Macro: {:.4f}\tRel: {:.4f}\tNo-Rel: {:.4f}'.format( \
            res_dict["ARI-Macro"], res_dict["Rel"], res_dict["No-Rel"])
    ACTC_msg = 'ACTC-Macro: {:.4f}\tMC: {:.4f}\tClaim: {:.4f}\tPremise: {:.4f}'.format( \
            res_dict["ACTC-Macro"], res_dict["MC"], res_dict["Claim"], res_dict["Premise"])
    ART_msg = 'ART-Macro: {:.4f}\tSup: {:.4f}\tAtc: {:.4f}'.format( \
        res_dict["ART-Macro"], res_dict["Sup"], res_dict["Atc"])
    macro_msg = 'Total-Macro: {:.4f}\tARI-Macro: {:.4f}\tACTC-Macro: {:.4f}\tART-Macro: {:.4f}'.format( \
            res_dict["Total-Macro"], res_dict["ARI-Macro"], res_dict["ACTC-Macro"], res_dict["ART-Macro"])
    return ARI_msg, ACTC_msg, ART_msg, macro_msg


def get_cdcp_eval_result(res_dict):
    ARI_msg = 'ARI-Macro: {:.4f}\tRel: {:.4f}\tNo-Rel: {:.4f}'.format( \
            res_dict["ARI-Macro"], res_dict["Rel"], res_dict["No-Rel"])
    ACTC_msg = 'ACTC-Macro: {:.4f}\tvalue: {:.4f}\tpolicy: {:.4f}\ttestimony: {:.4f}\tfact: {:.4f}\treference: {:.4f}'.format( \
            res_dict["ACTC-Macro"], res_dict["value"], res_dict["policy"], res_dict["testimony"], res_dict["fact"], res_dict["reference"])
    ART_msg = 'ART-Macro: {:.4f}\treason: {:.4f}\tevidence: {:.4f}'.format( \
        res_dict["ART-Macro"], res_dict["reason"], res_dict["evidence"])
    macro_msg = 'Total-Macro: {:.4f}\tARI-Macro: {:.4f}\tACTC-Macro: {:.4f}\tART-Macro: {:.4f}'.format( \
            res_dict["Total-Macro"], res_dict["ARI-Macro"], res_dict["ACTC-Macro"], res_dict["ART-Macro"])
    return ARI_msg, ACTC_msg, ART_msg, macro_msg


def preprocess_adj(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return torch.from_numpy(adj_normalized.A).float()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    # print(f"sparse adj: {adj}")
    rowsum = np.array(adj.sum(1))
    # print(f"rowsum: {rowsum.shape}")
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # print(d_mat_inv_sqrt)
    #  D^(-1/2)AD^(-1/2)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).tocoo()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = torch.Tensor([gamma])
        self.size_average = size_average
        if isinstance(alpha, (float, int, long)):
            if self.alpha > 1:
                raise ValueError('Not supported value, alpha should be small than 1.0')
            else:
                self.alpha = torch.Tensor([alpha, 1.0 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.alpha /= torch.sum(self.alpha)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # [N,C,H,W]->[N,C,H*W] ([N,C,D,H,W]->[N,C,D*H*W])
        # target
        # [N,1,D,H,W] ->[N*D*H*W,1]
        if self.alpha.device != input.device:
            self.alpha = torch.tensor(self.alpha, device=input.device)
        target = target.view(-1, 1)
        logpt = torch.log(input + 1e-10)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1, 1)
        pt = torch.exp(logpt)
        alpha = self.alpha.gather(0, target.view(-1))

        gamma = self.gamma

        if not self.gamma.device == input.device:
            gamma = torch.tensor(self.gamma, device=input.device)

        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss