def calc_pos_neg(gt_bboxes, pred_bboxes, threshold=0.5):
    pred_bboxes = sorted(pred_bboxes, reverse=True).copy()

    tps, fps, fns = 0, 0, 0
    for gt_bbox in gt_bboxes:
        best_iou = 0
        best_box = None
        for pred_bbox in pred_bboxes:
            iou = calc_iou(gt_bbox, pred_bbox)
            if iou > best_iou:
                best_iou = iou
                best_box = pred_bbox

        if best_iou > threshold:
            tps += 1
            pred_bboxes.remove(best_box)
        else:
            fns += 1

    fps += len(pred_bboxes)

    return tps, fps, fns


def calc_iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = bboxes1
    x21, y21, x22, y22 = bboxes2
    xA = max(x11, x21)
    yA = max(y11, y21)
    xB = min(x12, x22)
    yB = min(y12, y22)

    interArea = max((xB - xA + 1), 0) * max((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    return interArea / (boxAArea + boxBArea - interArea)


def f_score(val_preds, val_gts, eval_iou_threshold=0.3, beta=1):
    tps, fps, fns = 0, 0, 0
    for file in val_gts["image_id"].unique():
        gt_file = val_gts[val_gts["image_id"] == file]

        pred_file = val_preds[val_preds["image_id"] == file]

        tp, fp, fn = calc_pos_neg(
            gt_file[["y0", "x0", "y1", "x1"]].values.tolist(),
            pred_file[["y0", "x0", "y1", "x1"]].values.tolist(),
            eval_iou_threshold,
        )

        tps += tp
        fps += fp
        fns += fn

    f = ((1 + beta**2) * tps) / ((1 + beta**2) * tps + beta**2 * fns + fps)
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)

    return f, precision, recall
