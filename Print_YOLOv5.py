import torch

if __name__ == "__main__":
    yolov5l_ckpt = torch.load("yolov5l.pt")
    yolov5x_ckpt = torch.load("yolov5x.pt")
    yolov5l_keys = yolov5l_ckpt['model'].float().state_dict().keys()
    yolov5x_keys = yolov5x_ckpt['model'].float().state_dict().keys()

    for key1, key2 in zip(yolov5l_keys, yolov5x_keys):
        print(key1, key2)

    print("yolov5s.length = ", len(yolov5l_keys))
    print("yolov5x.length = ", len(yolov5x_keys))

    # torch.save(yolov5s_ckpt['model'].float().state_dict(), "yolov5s_state_dict.pt")
    # torch.save(yolov5x_ckpt['model'].float().state_dict(), "yolov5x_state_dict.pt")

    # torch.onnx.export(yolov5s_ckpt['model'], torch.rand(1, 3, 640, 640), "yolov5s.onnx")
