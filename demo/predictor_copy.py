# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch
import numpy as np
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        # 讀取 label 對應表
        label_mapping = {0: 0, 1: 4, 2: 3, 3: 2, 4: 4, 5: 3, 6: 4, 7: 2, 8: 0, 9: 2,
                        10: 4, 11: 2, 12: 5, 13: 5, 14: 7, 15: 5, 16: 6, 17: 6, 18: 6, 19: 4,
                        20: 5, 21: 2, 22: 2, 23: 2, 24: 3, 25: 4, 26: 4, 27: 6, 28: 6, 29: 4,
                        30: 6, 31: 3, 32: 7, 33: 4, 34: 5, 35: 5, 36: 5, 37: 5, 38: 1, 39: 1,
                        40: 4, 41: 4, 42: 4, 43: 1, 44: 1, 45: 4, 46: 4, 47: 4, 48: 4, 49: 5,
                        50: 3, 51: 6, 52: 6, 53: 8, 54: 3, 55: 4, 56: 0, 57: 5, 58: 4, 59: 6,
                        60: 4, 61: 4, 62: 6, 63: 5}
        
        self.GOOSE_9_stuff_colors = np.array([
                                        (169, 169, 169),  # 0: other
                                        (222, 136, 222),  # 1: artificial_structures
                                        (235, 255, 59),   # 2: artificial_ground
                                        (161, 136, 127),  # 3: natural_ground
                                        (255, 193, 7),    # 4: obstacle
                                        (244, 67, 54),    # 5: vehicle
                                        (76, 175, 80),    # 6: vegetation
                                        (143, 176, 255),  # 7: human
                                        (33, 150, 243)    # 8: sky
                                    ], dtype=np.uint8)
        
        self.GOOSE_64_stuff_colors = np.array([
                                                (0, 0, 0),         # 0: undefined
                                                (255, 255, 0),     # 1: traffic_cone
                                                (209, 87, 160),    # 2: snow
                                                (255, 52, 255),    # 3: cobble
                                                (255, 74, 70),     # 4: obstacle
                                                (0, 137, 65),      # 5: leaves
                                                (0, 111, 166),     # 6: street_light
                                                (163, 0, 89),      # 7: bikeway
                                                (255, 219, 229),   # 8: ego_vehicle
                                                (122, 73, 0),      # 9: pedestrian_crossing
                                                (0, 0, 166),       # 10: road_block
                                                (99, 255, 172),    # 11: road_marking
                                                (183, 151, 98),    # 12: car
                                                (0, 77, 67),       # 13: bicycle
                                                (143, 176, 255),   # 14: person
                                                (153, 125, 135),   # 15: bus
                                                (90, 0, 7),        # 16: forest
                                                (128, 150, 147),   # 17: bush
                                                (180, 168, 189),   # 18: moss
                                                (27, 68, 0),       # 19: traffic_light
                                                (79, 198, 1),      # 20: motorcycle
                                                (59, 93, 255),     # 21: sidewalk
                                                (74, 59, 83),      # 22: curb
                                                (255, 47, 128),    # 23: asphalt
                                                (97, 97, 90),      # 24: gravel
                                                (52, 54, 45),      # 25: boom_barrier
                                                (107, 121, 0),     # 26: rail_track
                                                (0, 194, 160),     # 27: tree_crown
                                                (255, 170, 146),   # 28: tree_trunk
                                                (136, 111, 76),    # 29: debris
                                                (0, 134, 237),     # 30: crops
                                                (209, 97, 0),      # 31: soil
                                                (221, 239, 255),   # 32: rider
                                                (0, 0, 53),        # 33: animal
                                                (123, 79, 75),     # 34: truck
                                                (161, 194, 153),   # 35: on_rails
                                                (48, 0, 24),       # 36: caravan
                                                (10, 166, 216),    # 37: trailer
                                                (1, 51, 73),       # 38: building
                                                (0, 132, 111),     # 39: wall
                                                (55, 33, 1),       # 40: rock
                                                (255, 181, 0),     # 41: fence
                                                (194, 255, 237),   # 42: guard_rail
                                                (160, 121, 191),   # 43: bridge
                                                (204, 7, 68),      # 44: tunnel
                                                (192, 185, 178),   # 45: pole
                                                (194, 255, 153),   # 46: traffic_sign
                                                (0, 30, 9),        # 47: misc_sign
                                                (190, 196, 89),    # 48: barrier_tape
                                                (111, 0, 98),      # 49: kick_scooter
                                                (12, 189, 102),    # 50: low_grass
                                                (238, 195, 255),   # 51: high_grass
                                                (69, 109, 117),    # 52: scenery_vegetation
                                                (183, 123, 104),   # 53: sky
                                                (122, 135, 161),   # 54: water
                                                (255, 140, 0),     # 55: wire
                                                (120, 141, 102),   # 56: outlier
                                                (250, 208, 159),   # 57: heavy_machinery
                                                (255, 138, 154),   # 58: container
                                                (232, 211, 23),    # 59: hedge
                                                (208, 208, 0),     # 60: barrel
                                                (221, 0, 0),       # 61: pipe
                                                (196, 164, 132),   # 62: tree_root
                                                (64, 64, 64)       # 63: military_vehicle
                                            ], dtype=np.uint8)


        # 建立映射陣列
        self.mapping_array = np.zeros(max(label_mapping.keys()) + 1, dtype=np.int32)
        for key, value in label_mapping.items():
            self.mapping_array[key] = value
    def run_on_image(self, image, save_9=False):
    # def run_on_image(self, image, save_9=True):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                mask = predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                # class_mask = (np.array(predictions["sem_seg"][24].to(self.cpu_device)) > 0.5).astype(np.uint8) * 255
                class_mask = None
                mask = np.array(mask)
                if save_9:
                    mask = self.mapping_array[mask]
                    vis_output = self.GOOSE_9_stuff_colors[mask]  # shape: (H, W, 3)
                    vis_output = Image.fromarray(vis_output)
                else:
                    # vis_output = self.GOOSE_64_stuff_colors[mask]  # shape: (H, W, 3) #ori 64
                    vis_output = self.GOOSE_9_stuff_colors[mask]
                    vis_output = Image.fromarray(vis_output)
                # vis_output = visualizer.draw_sem_seg(
                #     mask
                # )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        # return predictions, vis_output, mask
        return predictions, vis_output, mask, class_mask

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
