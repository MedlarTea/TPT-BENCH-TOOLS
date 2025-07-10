import json
import os
import os.path as osp

import math
import numpy as np
import argparse
import matplotlib.pyplot as plt

class TPTBenchEvaluator:
    def __init__(self, base_dir):
        self.data = {}

        print("------------ Reading Results --------------")
        self.init_dirs()
        # self.read_data()
        self.base_dir = os.path.join(base_dir, "baseline_results")
        self.GT_dir = os.path.join(base_dir, "GTs")

    def init_dirs(self, ):
        self.seq_names = [f"{idx:04d}" for idx in range(48)]

        self.FPS = {
            "RPF-ReID w/ R18": 20.3,
            "RPF-ReID+OCL w/ Parts-R18": 7.09,
            "RPF-ReID+OCL w/ R18": 18.5,
            "RPF-ReID w/ KPR": 11.5,
            "CARPE-ID w/ R18": 21.2,
            "CARPE-ID w/ KPR": 12.0,
            "Detection w/ KPR": 10.2,
            "Detection w/ R18": 20.7,
            "dimp50": 53.9,
            "keep_track": 28.6,
            "tamo": 12.9,
            "tomp101": 39.1,
            "stark": 36.4,
            "siamese_rpn": 55.4,
            "mixformer_convmae": 18.9,
            "mixformer_cvt": 35.9,
            "ltmu": 13.1,
            "siam_rcnn": 3.7,
        }

        self.paper_method = {
            "RPF-ReID w/ R18": "RPF-ReID w/ R18",
            "RPF-ReID+OCL w/ Parts-R18": "\makecell[l]{RPF-ReID+OCL w/ Parts-ResNet18\\\\citep{ye2024person}}",
            "RPF-ReID+OCL w/ R18": "\makecell[l]{RPF-ReID+OCL w/ R18\\\\citep{ye2024person}}",
            "RPF-ReID w/ KPR": "\makecell[l]{RPF-ReID w/ KPR\\\\citep{ye2024person}}",

            "CARPE-ID w/ R18": "CARPE-ID w/ R18",
            "CARPE-ID w/ KPR": "\makecell[l]{CARPE-ID w/ KPR\\\\citep{rollo2024icra}}",

            "Detection w/ KPR": "\makecell[l]{Detection w/ KPR\\\\citep{kpr}}",
            "Detection w/ R18": "\makecell[l]{Detection w/ R18\\\\citep{dendorfer2020mot20}}",

            "dimp50": "\makecell[l]{DiMP\\\\citep{dimp}}",
            "keep_track": "\makecell[l]{KeepTrack\\\\citep{keeptrack}}",
            "tamo": "\makecell[l]{TAMOs\\\\citep{tamo}}",
            "tomp101": "\makecell[l]{ToMP (ResNet101)\\\\citep{tomp}}",
            "stark": "\makecell[l]{STARK\\\\citep{stark}}",
            "siamese_rpn": "\makecell[l]{SiameseRPN++\\\\citep{li2019siamrpn++}}",
            "mixformer_convmae": "\makecell[l]{MixFormer-ConvMAE\\\\citep{mixformer2023tpami}}",
            "mixformer_cvt": "\makecell[l]{MixFormer-CVT\\\\citep{mixformer2022cvpr}}",
            "ltmu": "\makecell[l]{LTMU\\\\citep{ltmu}}",
            "siam_rcnn": "\makecell[l]{Siam-RCNN\\\\citep{siamrcnn}}",
        }

        self.methods = {
            "RPF-ReID w/ R18": "RPF-ReID-ResNet18",
            "RPF-ReID+OCL w/ Parts-R18": "RPF-ReID-OCL-Parts-ResNet18",
            "RPF-ReID+OCL w/ R18": "RPF-ReID-OCL-R18",
            "RPF-ReID w/ KPR": "RPF-ReID-KPR",
            "CARPE-ID w/ R18": "CARPE-ID-ResNet18",
            "CARPE-ID w/ KPR": "CARPE-ID-KPR",
            "Detection w/ KPR": "Detection-KPR",
            "Detection w/ R18": "Detection-ReNet18",
            "dimp50": "DiMP50",
            "keep_track": "KeepTrack",
            "tamo": "TAMO",
            "tomp101": "ToMP101",
            "stark": "STARK",
            "siamese_rpn": "SiameseRPN",
            "mixformer_convmae": "MixFormer-ConvMAE",
            "mixformer_cvt": "MixFormer-CVT",
            "ltmu": "LTMU",
            "siam_rcnn": "Siam-RCNN",
        }
    
    def read_data(self, ):
        for seq_name in self.seq_names:
            self.data[seq_name] = {}
            for method_name, file_name in self.methods.items():
                self.data[seq_name][method_name] = self.read_json(self.base_dir, seq_name, file_name)

    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        xA = max(x1, x1_)
        yA = max(y1, y1_)
        xB = min(x2, x2_)
        yB = min(y2, y2_)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box1Area = (x2 - x1 + 1) * (y2 - y1 + 1)
        box2Area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
        iou = interArea / float(box1Area + box2Area - interArea)
        return iou

    def return_box_with_highest_confidence(self, tracks):
        """
        track: [id, x1, y1, w, h, conf]
        """
        max_conf = 0
        max_box = []
        if len(tracks) == 0:
            return max_box, max_conf
        for track in tracks:
            if track[5] > max_conf:
                max_conf = track[5]
                max_box = [track[1], track[2], track[1]+track[3], track[2]+track[4]]
        return max_box, max_conf
    
    def return_maxRecall_at_100_precision(self, precision, recall):
        precision = np.array(precision)
        recall = np.array(recall)
        precision_100 = precision == 1.0
        if sum(precision_100) > 0:
            max_recall = max(recall[precision_100])
            return max_recall
        else:
            return -1
    
    def determine_thresholds(self, scores, resolution: int):
        """Determine thresholds for a given set of scores and a resolution. 
        The thresholds are determined by sorting the scores and selecting the thresholds that divide the sorted scores into equal sized bins. 
        
        Args:
            scores (Iterable[float]): Scores to determine thresholds for.
            resolution (int): Number of thresholds to determine.
            
        Returns:
            List[float]: List of thresholds.
        """
        scores = [score for score in scores if not math.isnan(score)] #and not score is None]
        scores = sorted(scores, reverse=True)

        if len(scores) > resolution - 2:
            delta = math.floor(len(scores) / (resolution - 2))
            idxs = np.round(np.linspace(delta, len(scores) - delta, num=resolution - 2)).astype(int)
            thresholds = [scores[idx] for idx in idxs]
        else:
            thresholds = scores

        thresholds.insert(0, math.inf)
        thresholds.insert(len(thresholds), -math.inf)

        return thresholds

    def plot_confidences(self, raw_result_list, plot_data_list):
        num_methods = len(raw_result_list)
        cols = 6
        rows = (num_methods + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(30, rows * 5))
        axs = axs.ravel()

        for i, (raw_result, plot_data) in enumerate(zip(raw_result_list, plot_data_list)):
            seq_name, method_name, result, frame_ids = raw_result

            confidence = result[:, 1]
            target_visible = result[:, 2]
            predicted_visible = result[:, 3] == True

            confidence_target_visible_1 = confidence[target_visible == 1]
            confidence_target_visible_0 = confidence[target_visible == 0]

            ax = axs[i]
            ax.hist(confidence_target_visible_1, bins=30, density=True, color='orange',
                    alpha=0.6, label='target visible')
            ax.hist(confidence_target_visible_0, bins=30, density=True, color='blue',
                    alpha=0.6, label='target non-visible')
            ax.set_title(f"{plot_data['method_name']} (F1: {max(plot_data['f1_scores']):.2f}, "
                        f"AP: {plot_data['ap']:.2f}, MR: {plot_data['avg_max_recall']:.2f})",
                        fontsize=10)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Density")
            ax.grid(True)
            ax.legend()

            fig_individual, ax_individual = plt.subplots(figsize=(6, 5))
            ax_individual.hist(confidence_target_visible_1, bins=30, density=True, color='orange',
                            alpha=0.6, label='target visible')
            ax_individual.hist(confidence_target_visible_0, bins=30, density=True, color='blue',
                            alpha=0.6, label='target non-visible')
            ax_individual.set_title(f"{plot_data['method_name']} (F-Score: {max(plot_data['f1_scores'])*100:.2f}, "
                                    f"AMR: {plot_data['avg_max_recall']*100:.2f})",
                                    fontsize=14)
            ax_individual.set_xlabel("Confidence", fontsize=14)
            ax_individual.set_ylabel("Noramlized Density", fontsize=14)
            ax_individual.legend(fontsize=16)

            output_dir = os.path.join(self.base_dir, "vis_results", seq_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            individual_file = os.path.join(output_dir, f"{self.save_name[plot_data['method_name']]}.png")
            plt.tight_layout()
            plt.savefig(individual_file, dpi=300)
            plt.close(fig_individual)

        for j in range(num_methods, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f"Evaluation Results for {seq_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        composite_file = os.path.join(self.base_dir, f"{seq_name}_all_methods_Confidence_eval.png")
        plt.savefig(composite_file, dpi=300)
        plt.close(fig)
                    
    def collect_result(self, seq_name, method_name):
        """Long-term Tracking Evaluation
        [1] Tracking for Half an Hour
        [2] Now you see me: evaluating performance in long-term visual tracking
        """
        result = []  # (iou, confidence, GT, PRED), GT and PRED -- 0 for not exist, 1 for exist
        frame_ids = []
        for i, frame_id in enumerate(self.data[seq_name][method_name].keys()):
            frame_ids.append(frame_id)
            is_exist = self.data[seq_name]["GT"][frame_id]["is_exist"]  # True for exist, False for not exist
            gt = self.data[seq_name]["GT"][frame_id]["bbox"]  # VISIBLE: [x1, y1, w, h]; NOT VISIBLE: [0, 0, 0, 0]
            gt_bbox = [gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]]
            pred = self.data[seq_name][method_name][frame_id]["target_info"]  # IDENTIFIED: [x1, y1, w, h, confidence]; NOT IDENTIFIED: [0, 0, 0, 0, -1]
            pred_bbox = [pred[0], pred[1], pred[0]+pred[2], pred[1]+pred[3]]
            pred_conf = pred[4]

            # the target person is visible
            if is_exist:
                if sum(pred_bbox) != 0:
                    iou = self.iou(gt_bbox, pred_bbox)
                    confidence = pred_conf if pred_conf != -1 else 1.0  # track initialization for MOT+ReID
                else:
                    ### predict from tracks ###
                    tracks = self.data[seq_name][method_name][frame_id]["tracks_target_conf_bbox"]
                    max_box, confidence = self.return_box_with_highest_confidence(tracks)
                    if len(max_box) == 0:
                        iou = 0
                    else:
                        iou = self.iou(gt_bbox, max_box) # binary
            # the target person is not visible
            else:
                if sum(pred_bbox) != 0:
                    iou = 0.0  # from [1]
                    confidence = pred_conf
                    if method_name == "RPF-ReID w/ KPR":
                        print(seq_name, frame_id, confidence)
                else:
                    ### predict from tracks ###
                    tracks = self.data[seq_name][method_name][frame_id]["tracks_target_conf_bbox"]
                    max_box, confidence = self.return_box_with_highest_confidence(tracks)
                    iou = 0

            result.append([iou, confidence, is_exist, sum(pred_bbox) != 0])  # [IOU, CONFIDENCE, GT_VISIBLE, PREDICTED_VISIBLE]
        
        result = np.array(result)
        return result, frame_ids
    
    def evaluate_with_LongSOT(self, result, seq_name, method_name, th_resolution=100):
        """We mainly calculate five metrics: precision, recall, f1_score, AO, average max recall at 100% precision
        # result: [IOU, CONFIDENCE, GT_VISIBLE, PRIDICTED_VISIBLE]
        return:
        """
        thresholds = self.determine_thresholds(result[:, 1], th_resolution)
        n_visible = sum(result[:, 2] == True)

        # print("n_visible:", n_visible)

        predicted_visible = result[:, 3] == True
        gt_visible = result[:, 2] == True

        # long-term tracking evaluation
        precision = len(thresholds) * [float(0)]
        recall = len(thresholds) * [float(0)]
        f_scores = len(thresholds) * [float(0)]
        ap = 0
        for i, threshold in enumerate(thresholds):
            # subset = (result[:, 1] >= threshold) * predicted_visible  # ignore absent targets
            subset = (result[:, 1] >= threshold)

            if np.sum(subset) == 0:
                precision[i] = 1
                recall[i] = 0
            else:
                precision[i] = np.mean(result[:, 0][subset])
                recall[i] = np.sum(result[:, 0][subset]) / n_visible
                if precision[i] > 1 or recall[i] > 1:
                    print(precision[i], recall[i])
            if i>1:
                ap += (recall[i] - recall[i-1]) * precision[i]
            
            f_scores[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-6)
        
        # short term tracking evaluation
        # result_visible = result[predicted_visible]
        result_visible = result[gt_visible]
        AO = np.mean(result_visible[:, 0])

        # max recall calculation
        iou_threshold = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        max_recalls = []
        for iou_th in iou_threshold:
            precision = len(thresholds) * [float(0)]
            recall = len(thresholds) * [float(0)]
            reuslt_binary = result.copy()
            reuslt_binary[:, 0] = reuslt_binary[:, 0] >= iou_th
            for i, threshold in enumerate(thresholds):
                subset = (result[:, 1] >= threshold)
                if np.sum(subset) == 0:
                    precision[i] = 1
                    recall[i] = 0
                else:
                    precision[i] = np.mean(reuslt_binary[:, 0][subset])
                    recall[i] = np.sum(reuslt_binary[:, 0][subset]) / n_visible
            max_recall = self.return_maxRecall_at_100_precision(precision, recall)
            max_recalls.append(max_recall)

        plot_data = {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "f1_scores": f_scores,
        "ap": ap,
        "ao": AO,
        "max_recalls": max_recalls,
        "avg_max_recall": np.mean(max_recalls),
        "seq_name": seq_name,
        "method_name": method_name
        }
        return plot_data
    
    def evaluate(self):
        result_avg = {}

        for i, seq_name in enumerate(self.seq_names):
            print("Evaluating {}/{}: {}".format(i, len(self.seq_names), seq_name))
            ### read data one by one for limited RAM ###
            self.data[seq_name] = {}
            for method_name, file_name in self.methods.items():
                self.data[seq_name][method_name] = self.read_json(osp.join(self.base_dir, seq_name, file_name+".json"))
            self.data[seq_name]["GT"] = self.read_json(osp.join(self.GT_dir, "{}.json".format(seq_name)))
            ### read data one by one for limited RAM ###

            plot_data_list = []
            raw_result_list = []
            for method_name in sorted(self.methods.keys()):
                result, frame_ids = self.collect_result(seq_name, method_name)
                raw_result_list.append([seq_name, method_name, result, frame_ids])

                plot_data = self.evaluate_with_LongSOT(result, seq_name, method_name, th_resolution=100)
                plot_data_list.append(plot_data) 
            
            # self.plot_confidences(raw_result_list, plot_data_list)\

            # initialize
            if i == 0:
                for j, raw_result in enumerate(raw_result_list):
                    seq_name, method_name, result, frame_ids = raw_result
                    plot_data = plot_data_list[j]
                    result_avg[plot_data['method_name']] = {}
                    result_avg[plot_data['method_name']] = {"f1":[], "ap":[], "ao":[], "avg_max_recall":[], "precision":[], "recall":[]}
            for j, raw_result in enumerate(raw_result_list):
                seq_name, method_name, result, frame_ids = raw_result
                plot_data = plot_data_list[j]

                max_idx = np.argmax(plot_data["f1_scores"])
                max_f1_score = plot_data["f1_scores"][max_idx]
                max_precision = plot_data["precision"][max_idx]
                max_recall = plot_data["recall"][max_idx]

                result_avg[plot_data['method_name']]["f1"].append(max_f1_score)
                result_avg[plot_data['method_name']]["precision"].append(max_precision)
                result_avg[plot_data['method_name']]["recall"].append(max_recall)
                result_avg[plot_data['method_name']]["ap"].append(plot_data["ap"])

                result_avg[plot_data['method_name']]["ao"].append(plot_data["ao"])

                result_avg[plot_data['method_name']]["avg_max_recall"].append(plot_data["avg_max_recall"])
            
            self.data[seq_name] = {}

        for method_name in result_avg.keys():
            print(f"{self.paper_method[method_name]} &{np.mean(result_avg[method_name]['ao'])*100:.2f} &{np.mean(result_avg[method_name]['f1'])*100:.2f} &{np.mean(result_avg[method_name]['avg_max_recall'])*100:.2f} &{self.FPS[method_name]} \\\\")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TPT Bench Evaluator')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory for the dataset')
    args = parser.parse_args()

    evaluator = TPTBenchEvaluator(args.dataset_dir)

    print("\n------------ Evaluating Results --------------")
    evaluator.evaluate()