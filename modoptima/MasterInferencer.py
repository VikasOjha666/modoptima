from deepsparse import Pipeline

class IYOLOv7:
	def __init__(self,filepath,iou_threshold,conf_threshold):
		self.filepath=filepath
        self.iou_threshold=iou_threshold
        self.conf_threshold=conf_threshold
		self.yolo_pipeline = Pipeline.create( task="yolo",model_path=model_stub,)

	def predict(self,path_or_img):
		return yolo_pipeline(images=path_or_img, iou_thres=0.4, conf_thres=0.20)

	def predict_and_draw(self,path_or_img):
		



