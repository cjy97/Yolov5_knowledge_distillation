import argparse
from pathlib import Path
from threading import Thread

import numpy as np
import yaml
from tqdm import tqdm
import torch

from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, \
    non_max_suppression, scale_coords, xywh2xyxy, set_logging, increment_path
from utils.metrics import ap_per_class, ConfusionMatrix, box_iou
from utils.plots import plot_images, output_to_target
from utils.torch_utils import time_sync
from slimming import load_pruners, LayerPruner


def test(data,
         model,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         single_cls=False,
         augment=False,
         verbose=False,
         plots=True, ):
    # Initialize/load model and set device
    # set_logging()
    device = next(model.parameters()).device  # get model device

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
    # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    # is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
    dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_sync()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_sync() - t

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_sync()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=[])
            t1 += time_sync() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            # path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, _ = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
    print(f"Results saved to {save_dir}")

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def main():
    set_logging()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load(opt.weights, map_location=torch.device('cpu'))  # load checkpoint
    if isinstance(model, dict):
        model = model['model'].float()
    model.model[-1].export = False

    pruners = load_pruners(opt.slimming_cfg)

    if len(pruners) == 2:
        layer_pruner = pruners[1]
        layer_pruner.set_model(model=model)
        selected = layer_pruner.compress(save_dir=save_dir)
        model_pruned_layer = layer_pruner.exec(model_path=save_dir / 'weights/layer_prune_model.pt')

        channel_pruner = pruners[0]
        channel_pruner.set_model(model=model_pruned_layer)
        selected = channel_pruner.compress()
        model_pruned = channel_pruner.exec(model_path=save_dir / 'weights/joint_prune_model.pt')
    else:
        pruner = pruners[0]
        if isinstance(pruner, LayerPruner):
            layer_pruner = pruner
            layer_pruner.set_model(model=model)
            selected = layer_pruner.compress(save_dir=save_dir)
            model_pruned = layer_pruner.exec(model_path=save_dir / 'weights/layer_prune_model.pt')
        else:
            channel_pruner = pruner
            channel_pruner.set_model(model=model)
            selected = channel_pruner.compress()
            model_pruned = channel_pruner.exec(model_path=save_dir / 'weights/filter_prune_model.pt')

    # test(data=opt.data, model=model_pruned.to(torch.device('cuda:0')), batch_size=opt.batch_size, plots=False)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='joint_prune.py')
    parser.add_argument('--weights', type=str,
                        default='yolov5l.pt')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')

    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')

    parser.add_argument('--project', default='runs/prune', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    ####################################################################################
    # Start Plugin arguments
    ####################################################################################
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="set flag to enable Automatic Mixed Precision (AMP). disabled by default ."
    )
    parser.add_argument(
        "--slimming-cfg",
        type=str, default='configs/YOLOv5l_Polar0.0005_Joint.yaml',
        help=""
    )
    ####################################################################################
    # End Plugin arguments
    ####################################################################################

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'weights').mkdir(parents=True, exist_ok=True)

    main()

    print('Done.')
