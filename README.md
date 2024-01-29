# YOLOv7-Rewrite

The code of the original [yolov7](https://github.com/WongKinYiu/yolov7) is difficult to use for real data sets. For example, if you want to use a metric other than mAP or if your images require special processing. All of this is hard to achieve in the original codebase.

Fortunately, this [repository](https://github.com/Chris-hughes10/Yolov7-training) contains a rewrite of the original YOLOv7 code. However, using the custom libraries makes the code a bit more complex again. For this reason, this fork was developed. This fork is a pure Pytorch version of the original code and uses the new `transforms.v2` detection library included in torchvision.

Everything but the absolutely necessary has been removed from the code. Essentially, this repository provides the model, the loss function and a possible metric. You can then write your own functionality and customize it according to your needs.

I tested this code on several private and public datasets. I was able to achieve at least as high or even higher mAP values than the original YOLOv7 code by customizing the code to my needs.

## Example: Car Object Detection

As an example, we will train YOLOv7 on the Kaggle dataset [Car Object Detection](https://www.kaggle.com/datasets/sshikamaru/car-object-detection). First, download this dataset and extract it so that the root directory contains `train_solution_bounding_boxes (1).csv` and a subdirectory `training_images`. Then the training can be started by executing `python example.py`. The Jupyter notebook `example_prediction.ipynb` shows how the model can be used for prediction.

Some additional information:
- `DetectionDataset`: requires that the dataset consists of a csv file with the columns `["image_path", "class_id", "xmin", "ymin", "xmax", "ymax"]`.
- The implemented metric is Precision@IOU, Recall@IOU, F@IOU, where IOU is the overlap. If IOU is greater than a certain threshold, the bounding box is considered a true positive. The classification error is not evaluated.
- All relevant parameters that were defined in the YAML files of the original YOLOv7 repository can now be found directly in the code. For example, the parameter `loss_ota` in the [original configuration file](https://github.com/WongKinYiu/yolov7/blob/main/data/hyp.scratch.custom.yaml) can be deactivated by setting `ota_loss=False` in `example.py`.
- You have to implement certain functions yourself if you need them. The current code provides a good basis that can still be improved.

## Acknowledgements

- Original YOLO implementation: [yolov7](https://github.com/WongKinYiu/yolov7)
- Rewritten code: [Yolov7-training](https://github.com/Chris-hughes10/Yolov7-training)