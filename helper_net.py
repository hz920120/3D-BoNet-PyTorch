import torch


def cast(x, dtype):
    x = x.type(dtype)
    return x


class Ops:

    @staticmethod
    def cast(x, dtype):
        x = x.type(dtype)
        return x

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return torch.nn.ReLU(x)

    @staticmethod
    def xxlu(x, label, name=None):
        if label == 'relu':
            return Ops.relu(x)
        if label == 'lrelu':
            return Ops.lrelu(x, leak=0.2)

    ####################################
    @staticmethod
    def gather_tensor_along_2nd_axis(bat_bb_pred, bat_bb_indices):
        bat_size = bat_bb_pred.size()[0]
        [_, ins_max_num, d1, d2] = bat_bb_pred.shape
        bat_size_range = torch.arange(bat_size, device=torch.device('cuda'))
        bat_size_range_flat = torch.reshape(bat_size_range, [-1, 1])
        bat_size_range_flat_repeat = Ops.repeat(bat_size_range_flat, [1, int(ins_max_num)])
        bat_size_range_flat_repeat = torch.reshape(bat_size_range_flat_repeat, [-1])

        indices_2d_flat = torch.reshape(bat_bb_indices, [-1])
        indices_2d_flat_repeat = bat_size_range_flat_repeat * int(ins_max_num) + indices_2d_flat

        bat_bb_pred = torch.reshape(bat_bb_pred, [-1, int(d1), int(d2)])
        # TODO bat_bb_pred_new = tf.gather(bat_bb_pred, indices_2d_flat_repeat)
        # repeat = torch.from_numpy(indices_2d_flat_repeat).long()
        # repeat = repeat.unsqueeze(1)
        # repeat = repeat.expand(1,3,5)
        # repeat = repeat.squeeze()
        indices_2d_flat_repeat = indices_2d_flat_repeat.unsqueeze(1).expand(indices_2d_flat_repeat.size()[0],
                                                                            bat_bb_pred.size()[1])
        indices_2d_flat_repeat = indices_2d_flat_repeat.unsqueeze(2).expand(indices_2d_flat_repeat.size()[0],
                                                                            bat_bb_pred.size()[1],
                                                                            bat_bb_pred.size()[2])
        bat_bb_pred_new = bat_bb_pred.gather(0, indices_2d_flat_repeat)
        bat_bb_pred_new = torch.reshape(bat_bb_pred_new, [bat_size, int(ins_max_num), int(d1), int(d2)])

        return bat_bb_pred_new

    # @staticmethod
    # def hungarian(loss_matrix, bb_gt):
    #     box_mask = np.array([[0, 0, 0], [0, 0, 0]])
    #
    #     def assign_mappings_valid_only(cost, gt_boxes):
    #         # return ordering : batch_size x num_instances
    #         loss_total = 0.
    #         batch_size, num_instances = cost.shape[:2]
    #         ordering = np.zeros(shape=[batch_size, num_instances]).astype(np.int32)
    #         for idx in range(batch_size):
    #             ins_gt_boxes = gt_boxes[idx]
    #             ins_count = 0
    #             for box in ins_gt_boxes:
    #                 if np.array_equal(box, box_mask):
    #                     break
    #                 else:
    #                     ins_count += 1
    #             valid_cost = cost[idx][:ins_count]
    #             row_ind, col_ind = linear_sum_assignment(valid_cost)
    #             unmapped = num_instances - ins_count
    #             if unmapped > 0:
    #                 rest = np.array(range(ins_count, num_instances))
    #                 row_ind = np.concatenate([row_ind, rest])
    #                 unmapped_ind = np.array(list(set(range(num_instances)) - set(col_ind)))
    #                 col_ind = np.concatenate([col_ind, unmapped_ind])
    #
    #             loss_total += cost[idx][row_ind, col_ind].sum()
    #             ordering[idx] = np.reshape(col_ind, [1, -1])
    #         return ordering, (loss_total / float(batch_size * num_instances)).astype(np.float32)
    #
    #     ###### TODO tf.py_func equivalent in pytorch?
    #     ordering, loss_total = tf.py_func(assign_mappings_valid_only, [loss_matrix, bb_gt], [torch.int32, torch.float32])
    #     ordering, loss_total = torch.ones(4 * 24, device='cuda'), torch.tensor(1.0, device='cuda')
    #     return ordering, loss_total

    @staticmethod
    def bbvert_association(X_pc, y_bbvert_pred, Y_bbvert, label=''):
        points_num = X_pc.size()[1]
        bbnum = int(y_bbvert_pred.shape[1])
        points_xyz = X_pc[:, :, 0:3]
        points_xyz = Ops.repeat(points_xyz[:, None, :, :], [1, bbnum, 1, 1])

        ##### get points hard mask in each gt bbox
        gt_bbox_min_xyz = Y_bbvert[:, :, 0, :]
        gt_bbox_max_xyz = Y_bbvert[:, :, 1, :]
        gt_bbox_min_xyz = gt_bbox_min_xyz[:, :, None, :].repeat([1, 1, points_num, 1])
        gt_bbox_max_xyz = gt_bbox_max_xyz[:, :, None, :].repeat([1, 1, points_num, 1])
        tp1_gt = gt_bbox_min_xyz - points_xyz
        tp2_gt = points_xyz - gt_bbox_max_xyz
        tp_gt = tp1_gt * tp2_gt
        mean = torch.mean(cast(torch.ge(tp_gt, 0.), torch.float32), dim=-1)
        points_in_gt_bbox_prob = cast(torch.eq(mean, torch.ones(mean.shape, device=torch.device('cuda'))), torch.float32)

        ##### get points soft mask in each pred bbox ---> Algorithm 1
        pred_bbox_min_xyz = y_bbvert_pred[:, :, 0, :]
        pred_bbox_max_xyz = y_bbvert_pred[:, :, 1, :]
        pred_bbox_min_xyz = Ops.repeat(pred_bbox_min_xyz[:, :, None, :], [1, 1, points_num, 1])
        pred_bbox_max_xyz = Ops.repeat(pred_bbox_max_xyz[:, :, None, :], [1, 1, points_num, 1])
        tp1_pred = pred_bbox_min_xyz - points_xyz
        tp2_pred = points_xyz - pred_bbox_max_xyz
        tp_pred = 100 * tp1_pred * tp2_pred
        tp_pred = torch.max(torch.min(tp_pred, torch.empty(tp_pred.shape, device=torch.device('cuda')).fill_(20.)),
                              torch.empty(tp_pred.shape, device=torch.device('cuda')).fill_(-20.0))
        points_in_pred_bbox_prob = 1.0 / (1.0 + torch.exp(-1.0 * tp_pred))
        points_in_pred_bbox_prob = torch.min(points_in_pred_bbox_prob, dim=-1).values

        ##### get bbox cross entropy scores
        prob_gt = Ops.repeat(points_in_gt_bbox_prob[:, :, None, :], [1, 1, bbnum, 1])
        prob_pred = Ops.repeat(points_in_pred_bbox_prob[:, None, :, :], [1, bbnum, 1, 1])
        ce_scores_matrix = - prob_gt * torch.log(prob_pred + 1e-8) - (1 - prob_gt) * torch.log(1 - prob_pred + 1e-8)
        ce_scores_matrix = torch.mean(ce_scores_matrix, dim=-1)

        ##### get bbox soft IOU
        TP = torch.sum(prob_gt * prob_pred, dim=-1)
        FP = torch.sum(prob_pred, dim=-1) - TP
        FN = torch.sum(prob_gt, dim=-1) - TP
        iou_scores_matrix = TP / (TP + FP + FN + 1e-6)
        # iou_scores_matrix = 1.0/iou_scores_matrix  # bad, don't use
        iou_scores_matrix = -1.0 * iou_scores_matrix  # to minimize

        ##### get bbox l2 scores
        l2_gt = Ops.repeat(Y_bbvert[:, :, None, :, :], [1, 1, bbnum, 1, 1])
        l2_pred = Ops.repeat(y_bbvert_pred[:, None, :, :, :], [1, bbnum, 1, 1, 1])
        l2_gt = torch.reshape(l2_gt, [-1, bbnum, bbnum, 2 * 3])
        l2_pred = torch.reshape(l2_pred, [-1, bbnum, bbnum, 2 * 3])
        l2_scores_matrix = torch.mean((l2_gt - l2_pred) ** 2, dim=-1)

        ##### bbox association
        if label == 'use_all_ce_l2_iou':
            associate_maxtrix = ce_scores_matrix + l2_scores_matrix + iou_scores_matrix
        elif label == 'use_both_ce_l2':
            associate_maxtrix = ce_scores_matrix + l2_scores_matrix
        elif label == 'use_both_ce_iou':
            associate_maxtrix = ce_scores_matrix + iou_scores_matrix
        elif label == 'use_both_l2_iou':
            associate_maxtrix = l2_scores_matrix + iou_scores_matrix
        elif label == 'use_only_ce':
            associate_maxtrix = ce_scores_matrix
        elif label == 'use_only_l2':
            associate_maxtrix = l2_scores_matrix
        elif label == 'use_only_iou':
            associate_maxtrix = iou_scores_matrix
        else:
            associate_maxtrix = None
            print('association label error!');
            exit()

        ######
        # hun = Hungarian()
        # pred_bborder, association_score_min = hun(associate_maxtrix, Y_bbvert)
        # # pred_bborder, association_score_min = Ops.hungarian(associate_maxtrix, bb_gt=Y_bbvert)
        # pred_bborder = cast(pred_bborder, dtype=torch.int32)
        # y_bbvert_pred_new = Ops.gather_tensor_along_2nd_axis(y_bbvert_pred, pred_bborder)
        #
        # return y_bbvert_pred_new, pred_bborder
        return associate_maxtrix, Y_bbvert

    # @staticmethod
    # def tile(tensor, reps):
    #     tensor_numpy = tensor.cpu().detach().numpy()
    #     tensor_numpy = np.tile(tensor_numpy, reps)
    #     res = torch.from_numpy(tensor_numpy)
    #     return res.cuda()

    # def tile(a, dim, n_tile):
    #     init_dim = a.size(dim)
    #     repeat_idx = [1] * a.dim()
    #     repeat_idx[dim] = n_tile
    #     a = a.repeat(*(repeat_idx))
    #     order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    #     return torch.index_select(a, dim, order_index)

    @staticmethod
    def repeat(tensor, dims):
        if len(dims) != len(tensor.shape):
            raise ValueError("The length of the second argument must equal the number of dimensions of the first.")
        for index, dim in enumerate(dims):
            repetition_vector = [1] * (len(dims) + 1)
            repetition_vector[index + 1] = dim
            new_tensor_shape = list(tensor.shape)
            new_tensor_shape[index] *= dim
            tensor = tensor.unsqueeze(index + 1).repeat(repetition_vector).reshape(new_tensor_shape)
        return tensor

    # @staticmethod
    # def maximum(tensor1, tensor2):
    #     # tensor1_numpy = tensor1.cpu().detach().numpy()
    #     # tensor2_numpy = tensor2.cpu().detach().numpy()
    #     # tensor_numpy = np.maximum(tensor1_numpy, tensor2_numpy)
    #     # res = torch.from_numpy(tensor_numpy)
    #     res = torch.max(tensor1, tensor2, keepdim=True)
    #     return res.cuda()

    # @staticmethod
    # def minimum(tensor1, tensor2):
    #     tensor1_numpy = tensor1.cpu().detach().numpy()
    #     tensor2_numpy = tensor2.cpu().detach().numpy()
    #     tensor_numpy = np.minimum(tensor1_numpy, tensor2_numpy)
    #     res = torch.from_numpy(tensor_numpy)
    #     return res.cuda()

    @staticmethod
    def bbscore_association(y_bbscore_pred_raw, pred_bborder):
        y_bbscore_pred_raw = y_bbscore_pred_raw[:, :, None, None]
        y_bbscore_pred_new = Ops.gather_tensor_along_2nd_axis(y_bbscore_pred_raw, pred_bborder)

        y_bbscore_pred_new = torch.reshape(y_bbscore_pred_new, [-1, int(y_bbscore_pred_new.shape[1])])
        return y_bbscore_pred_new

    ####################################  sem loss
    @staticmethod
    def get_loss_psem_ce(inputs, targets):
        p = torch.softmax(inputs, dim=-1, dtype=torch.float32)
        h = -targets * torch.log(p)
        res = torch.sum(h, dim=-1, dtype=torch.float32)
        return torch.mean(res, dtype=torch.float32)

    ####################################  bbox loss
    @staticmethod
    def get_loss_bbvert(X_pc, y_bbvert_pred, Y_bbvert, label=''):
        points_num = X_pc.size()[1]
        bb_num = int(Y_bbvert.shape[1])
        points_xyz = X_pc[:, :, 0:3]
        points_xyz = Ops.repeat(points_xyz[:, None, :, :], [1, bb_num, 1, 1])

        ##### get points hard mask in each gt bbox
        gt_bbox_min_xyz = Y_bbvert[:, :, 0, :]
        gt_bbox_max_xyz = Y_bbvert[:, :, 1, :]
        gt_bbox_min_xyz = Ops.repeat(gt_bbox_min_xyz[:, :, None, :], [1, 1, points_num, 1])
        gt_bbox_max_xyz = Ops.repeat(gt_bbox_max_xyz[:, :, None, :], [1, 1, points_num, 1])
        tp1_gt = gt_bbox_min_xyz - points_xyz
        tp2_gt = points_xyz - gt_bbox_max_xyz
        tp_gt = tp1_gt * tp2_gt
        points_in_gt_bbox_prob = cast(
            torch.eq(torch.mean(cast(torch.ge(tp_gt, 0.), torch.float32), dim=-1), torch.tensor([1.0], device=torch.device('cuda'))),
            torch.float32)

        ##### get points soft mask in each pred bbox
        pred_bbox_min_xyz = y_bbvert_pred[:, :, 0, :]
        pred_bbox_max_xyz = y_bbvert_pred[:, :, 1, :]
        pred_bbox_min_xyz = Ops.repeat(pred_bbox_min_xyz[:, :, None, :], [1, 1, points_num, 1])
        pred_bbox_max_xyz = Ops.repeat(pred_bbox_max_xyz[:, :, None, :], [1, 1, points_num, 1])
        tp1_pred = pred_bbox_min_xyz - points_xyz
        tp2_pred = points_xyz - pred_bbox_max_xyz
        tp_pred = 100 * tp1_pred * tp2_pred
        tp_pred = torch.max(torch.min(tp_pred, torch.empty(tp_pred.shape, device=torch.device('cuda')).fill_(20.)),
                              torch.empty(tp_pred.shape, device=torch.device('cuda')).fill_(-20.0))
        # tp_pred = torch.maximum(torch.minimum(tp_pred, torch.tensor(20.0)), torch.tensor(-20.0))
        points_in_pred_bbox_prob = 1.0 / (1.0 + torch.exp(-1.0 * tp_pred))
        points_in_pred_bbox_prob = torch.min(points_in_pred_bbox_prob, dim=-1).values

        ##### helper -> the valid bbox (the gt boxes are zero-padded during data processing, pickup valid ones here)
        Y_bbox_helper = torch.sum(torch.reshape(Y_bbvert, [-1, bb_num, 6]), dim=-1)
        Y_bbox_helper = cast(torch.gt(Y_bbox_helper, 0.), torch.float32)

        ##### 1. get ce loss of valid/positive bboxes, don't count the ce_loss of invalid/negative bboxes
        Y_bbox_helper_tp1 = Ops.repeat(Y_bbox_helper[:, :, None], [1, 1, points_num])
        bbox_loss_ce_all = -points_in_gt_bbox_prob * torch.log(points_in_pred_bbox_prob + (1e-8)) \
                           - (1. - points_in_gt_bbox_prob) * torch.log(1. - points_in_pred_bbox_prob + 1e-8)
        bbox_loss_ce_pos = torch.sum(bbox_loss_ce_all * Y_bbox_helper_tp1) / torch.sum(Y_bbox_helper_tp1)
        bbox_loss_ce = bbox_loss_ce_pos

        ##### 2. get iou loss of valid/positive bboxes
        TP = torch.sum(points_in_pred_bbox_prob * points_in_gt_bbox_prob, dim=-1)
        FP = torch.sum(points_in_pred_bbox_prob, dim=-1) - TP
        FN = torch.sum(points_in_gt_bbox_prob, dim=-1) - TP
        bbox_loss_iou_all = TP / (TP + FP + FN + 1e-6)
        bbox_loss_iou_all = -1.0 * bbox_loss_iou_all
        bbox_loss_iou_pos = torch.sum(bbox_loss_iou_all * Y_bbox_helper) / torch.sum(Y_bbox_helper)
        bbox_loss_iou = bbox_loss_iou_pos

        ##### 3. get l2 loss of both valid/positive bboxes
        bbox_loss_l2_all = (Y_bbvert - y_bbvert_pred) ** 2
        bbox_loss_l2_all = torch.mean(torch.reshape(bbox_loss_l2_all, [-1, bb_num, 6]), dim=-1)
        bbox_loss_l2_pos = torch.sum(bbox_loss_l2_all * Y_bbox_helper) / torch.sum(Y_bbox_helper)

        ## to minimize the 3D volumn of invalid/negative bboxes, it serves as a regularizer to penalize false pred bboxes
        ## it turns out to be quite helpful, but not discussed in the paper
        bbox_pred_neg = Ops.repeat((1. - Y_bbox_helper)[:, :, None, None], [1, 1, 2, 3]) * y_bbvert_pred
        bbox_loss_l2_neg = (bbox_pred_neg[:, :, 0, :] - bbox_pred_neg[:, :, 1, :]) ** 2
        bbox_loss_l2_neg = torch.sum(bbox_loss_l2_neg) / (torch.sum(1. - Y_bbox_helper) + 1e-8)

        bbox_loss_l2 = bbox_loss_l2_pos + bbox_loss_l2_neg

        #####
        if label == 'use_all_ce_l2_iou':
            bbox_loss = bbox_loss_ce + bbox_loss_l2 + bbox_loss_iou
        elif label == 'use_both_ce_l2':
            bbox_loss = bbox_loss_ce + bbox_loss_l2
        elif label == 'use_both_ce_iou':
            bbox_loss = bbox_loss_ce + bbox_loss_iou
        elif label == 'use_both_l2_iou':
            bbox_loss = bbox_loss_l2 + bbox_loss_iou
        elif label == 'use_only_ce':
            bbox_loss = bbox_loss_ce
        elif label == 'use_only_l2':
            bbox_loss = bbox_loss_l2
        elif label == 'use_only_iou':
            bbox_loss = bbox_loss_iou
        else:
            bbox_loss = None
            print('bbox loss label error!')
            exit()

        return bbox_loss, bbox_loss_l2, bbox_loss_ce, bbox_loss_iou

    @staticmethod
    def get_loss_bbscore(y_bbscore_pred, Y_bbvert):
        bb_num = int(Y_bbvert.shape[1])

        ##### helper -> the valid bbox
        Y_bbox_helper = torch.sum(torch.reshape(Y_bbvert, [-1, bb_num, 6]), dim=-1)
        Y_bbox_helper = cast(torch.gt(Y_bbox_helper, 0.), torch.float32)

        ##### bbox score loss
        bbox_loss_score = torch.mean(-Y_bbox_helper * torch.log(y_bbscore_pred + 1e-8)
                                     - (1. - Y_bbox_helper) * torch.log(1. - y_bbscore_pred + 1e-8))
        return bbox_loss_score

    ####################################  pmask loss
    @staticmethod
    def get_loss_pmask(X_pc, y_pmask_pred, Y_pmask):
        y_pmask_pred = y_pmask_pred*0
        points_num = X_pc.size()[1]
        ##### valid ins
        Y_pmask_helper = torch.sum(Y_pmask, dim=-1)
        Y_pmask_helper = cast(torch.gt(Y_pmask_helper, 0.), torch.float32)
        Y_pmask_helper = Y_pmask_helper[:, :, None].repeat([1, 1, points_num])

        Y_pmask = Y_pmask * Y_pmask_helper
        y_pmask_pred = y_pmask_pred * Y_pmask_helper

        ##### focal loss
        alpha = 0.75
        gamma = 2
        pmask_loss_focal_all = -Y_pmask * alpha * ((1. - y_pmask_pred) ** gamma) * torch.log(y_pmask_pred + 1e-8) \
                               - (1. - Y_pmask) * (1. - alpha) * (y_pmask_pred ** gamma) * torch.log(
            1. - y_pmask_pred + 1e-8)
        pmask_loss_focal = torch.sum(pmask_loss_focal_all * Y_pmask_helper) / torch.sum(Y_pmask_helper)

        ## the above "alpha" makes the loss to be small
        ## then use a constant, so it's numerically comparable with other losses (e.g., semantic loss, bbox loss)
        pmask_loss = 30 * pmask_loss_focal

        return pmask_loss
