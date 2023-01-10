from deepsparse import Pipeline

class IYOLOv7:
    def __init__(self,filepath,iou_threshold=0.4,conf_threshold=0.20):
        """
        Initializes the inference pipeline.

        params:
         filepath: path to the saved onnx file.
         iou_threshold: IOU threshold to perform NMS.
         conf_threshold: Confidence score to filter out boxes with NMS

        """
        self.filepath=filepath
        self.iou_threshold=iou_threshold
        self.conf_threshold=conf_threshold
        self.yolo_pipeline = Pipeline.create( task="yolo",model_path=self.filepath,)

    def predict(self,path_or_img):
         """
         Runs the prediction on the model.
         
         params:
           path_or_img: Path to the image or the numpy representation of image.

         returns:
           boxes: Bounding boxes.
           scores: Confidence score of the model.
           labels: Labels of the predicted class.


         """
         pred=self.yolo_pipeline(images=path_or_img, iou_thres=self.iou_threshold, conf_thres=self.iou_threshold)
         boxes=pred.boxes
         scores=pred.scores
         labels=pred.scores

         return boxes,scores,labels

    





