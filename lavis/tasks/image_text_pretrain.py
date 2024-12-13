"""
 Copyright (c) 2024, Li Guo.
 All rights reserved.
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import torch
import logging
import contextlib
import torch
import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, main_process, is_main_process
from lavis.common.logger import MetricLogger
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import os 
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
from datetime import datetime
import traceback
from chexpert_labeler.loader import Loader
from chexpert_labeler.stages.extract import Extractor
from chexpert_labeler.stages.classify import Classifier
from chexpert_labeler.stages.aggregate import Aggregator
from chexpert_labeler.constants import CATEGORIES
from chexpert_labeler import write
from pycocoevalcap.meteor.meteor import Meteor

@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass


@registry.register_task("image_text_pretrain_ce")
class ImageTextPretrainTask_CE(BaseTask):
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chexpert_args = None
        self._initialize_chexpert_components()

    def _initialize_chexpert_components(self):
        """Initialize CheXpert labeler components"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
        chexpert_dir = os.path.join(root_dir, "chexpert_labeler")
        
        # Set up paths
        self.mention_dir = Path(os.path.join(chexpert_dir, "phrases", "mention"))
        self.unmention_dir = Path(os.path.join(chexpert_dir, "phrases", "unmention"))
        patterns_dir = Path(os.path.join(chexpert_dir, "patterns"))
        
        # Initialize components once
        self.extractor = Extractor(
            mention_phrases_dir=self.mention_dir,
            unmention_phrases_dir=self.unmention_dir,
            verbose=False
        )
        
        self.classifier = Classifier(
            pre_negation_uncertainty_path=os.path.join(patterns_dir, "new_pre_negation_uncertainty.txt"),
            negation_path=os.path.join(patterns_dir, "new_negation.txt"),
            post_negation_uncertainty_path=os.path.join(patterns_dir, "new_post_negation_uncertainty.txt"),
            verbose=False
        )
        
        self.aggregator = Aggregator(
            categories=CATEGORIES,
            verbose=False
        )
        
    def label_reports(self, reports_path, output_path):
        """Custom labeling function using initialized components"""
        
        # Create loader for current reports
        loader = Loader(
            reports_path=reports_path,
            sections_to_extract=['findings'],
            extract_strict=True
        )
        
        # Process reports
        loader.load()
        self.extractor.extract(loader.collection)
        self.classifier.classify(loader.collection)
        labels = self.aggregator.aggregate(loader.collection)
        
        # Write results
        write(loader.reports, labels, output_path, verbose=False)


    def valid_step(self, model, samples):
        """
        Validation step that computes ITC and ITM losses without gradients
        """
        with torch.no_grad():
            with self.maybe_autocast():
                output = model.generate(samples)

        return {
            # "itc_loss": output.loss_itc.item(),
            # "itm_loss": output.loss_itm.item(),
            "predicted_captions": output.predicted_captions,
            
            "gt_captions": output.gt_captions,
            
            "study_id": samples["study_id"]
        }
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    def get_scorer(self):
        scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        return scorers
    
    def prepare_chexpert_csv(self, reports, filename):
        """
        Prepare reports in CheXpert format
        """
        processed_reports = []
        for report in reports:
            report = report.strip()
            processed_report = f"findings:\n{report}"
            processed_reports.append(processed_report)
        
        df = pd.DataFrame({'Report': processed_reports})
        df.to_csv(filename, index=False,header=False, encoding='utf-8')
        return filename

    
    @main_process
    def compute_clinical_metrics(self, pred_reports, gt_reports, all_study_id):
        """Calculate both micro-averaged and per-category clinical metrics"""
        clinical_metrics = {}
        temp_files = {}
        
        try:
            # Create temporary files with unique names
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_files = {
                'pred_csv': f"predictions_{timestamp}.csv",
                'gt_csv': f"ground_truths_{timestamp}.csv",
                'pred_output': f"pred_labeled_{timestamp}.csv",
                'gt_output': f"gt_labeled_{timestamp}.csv"
            }
            
            # Prepare CSV files
            self.prepare_chexpert_csv(pred_reports, temp_files['pred_csv'])
            self.prepare_chexpert_csv(gt_reports, temp_files['gt_csv'])
            
            print("Labeling predictions...")
            self.label_reports(temp_files['pred_csv'], temp_files['pred_output'])
            print("Labeling ground truth...")
            self.label_reports(temp_files['gt_csv'], temp_files['gt_output'])
            
            # Read labeled results
            pred_labels = pd.read_csv(temp_files['pred_output'])
            gt_labels = pd.read_csv(temp_files['gt_output'])
            
            # Get labels (excluding the Report column)
            pred_labels = pred_labels.iloc[:, 1:].values
            gt_labels = gt_labels.iloc[:, 1:].values
            
            # Create analysis table
            if is_main_process():
                analysis_df = self.create_analysis_table(
                    pred_reports,
                    gt_reports,
                    all_study_id,
                    pred_labels,
                    gt_labels
                )
                
                if analysis_df is not None:
                    print("\nAnalysis Summary:")
                    print(f"Total samples analyzed: {len(analysis_df)}")
                    print(f"Average match percentage: {analysis_df['match_percentage'].mean():.2f}%")
                    print("\nExample Analysis (last 2 rows):")
                    print(analysis_df.tail(2).to_string())
            
            # Initialize counters for micro averaging
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            # Print header for per-category results
            print("\n" + "="*130)
            print(f"{'Category':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
            print("="*130)
            
            # Process each category
            for i, category in enumerate(CATEGORIES):
                category_preds = pred_labels[:, i].copy()
                category_truths = gt_labels[:, i].copy()
                
                # Convert NaN and -1 to 0
                category_preds[np.isnan(category_preds)] = 0
                category_preds[category_preds == -1] = 0
                category_truths[np.isnan(category_truths)] = 0
                category_truths[category_truths == -1] = 0
                
                # Calculate confusion matrix for this category
                conf_matrix = confusion_matrix(category_truths, category_preds)
                tn, fp, fn, tp = conf_matrix.ravel()
                
                # Accumulate values for micro averaging
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                # Calculate per-category metrics
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics = {
                    f"{category}_accuracy": accuracy,
                    f"{category}_precision": precision,
                    f"{category}_recall": recall,
                    f"{category}_f1": f1
                }
                clinical_metrics.update(metrics)
                
                # Count positives in ground truth for support
                support = np.sum(category_truths == 1)
                
                # Print per-category results
                print(f"{category:<25} {accuracy:>11.4f} {precision:>11.4f} {recall:>11.4f} {f1:>11.4f} {support:>11d}")
                
                # Print confusion matrix
                print(f"\nConfusion Matrix for {category}:")
                print(f"TN={tn}, FP={fp}")
                print(f"FN={fn}, TP={tp}")
                print("-"*130)
            
            # Calculate micro-averaged metrics
            # Micro-precision only uses TP and FP
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            # Micro-recall uses TP and FN
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            # Micro-F1 uses both
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            
            micro_metrics = {
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1
            }
            clinical_metrics.update(micro_metrics)
            
            # Print micro-averaged results with calculation details
            print("\n" + "="*130)
            print("MICRO-AVERAGED RESULTS:")
            print(f"\nCalculation details:")
            print(f"Total TP: {total_tp}")
            print(f"Total FP: {total_fp}")
            print(f"Total FN: {total_fn}")
            print(f"\nMicro-precision = TP / (TP + FP) = {total_tp} / ({total_tp} + {total_fp}) = {micro_precision:.4f}")
            print(f"Micro-recall = TP / (TP + FN) = {total_tp} / ({total_tp} + {total_fn}) = {micro_recall:.4f}")
            print(f"Micro-F1 = 2 * (precision * recall) / (precision + recall) = {micro_f1:.4f}")
            print("="*130)
                
        except Exception as e:
            print(f"Error in clinical metrics computation: {e}")
            traceback.print_exc()
        
        finally:
            # Clean up temporary files
            for file_path in temp_files.values():
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")
        
        return clinical_metrics
    
    @main_process
    def compute_metrics(self, Qf_predictions, ground_truths):
        val_metrics = {}
        # Prepare data for NLP metrics calculation
        gts = {}  # ground truth
        Qf_res = {}  # Q-former predictions
        valid_samples = 0
        empty_samples = 0

        for i, (Qf_pred, gt) in enumerate(zip(Qf_predictions, ground_truths)):
            if not Qf_pred or not gt:
                empty_samples += 1
                continue
                
            try:
                gts[valid_samples] = [gt]
                Qf_res[valid_samples] = [Qf_pred]
                # LLM_res[valid_samples] = [LLM_pred]
                valid_samples += 1
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                empty_samples += 1
                continue

        if valid_samples > 0:
            try:
                # Calculate BLEU scores
                Qf_eval_results = self.compute_scores(gts, Qf_res)
                
                # update metric
                Qf_eval_results = {"Q_" + k: v for k, v in Qf_eval_results.items()}
                val_metrics.update(Qf_eval_results)

                metrics_str = "Q-former |"
                for metric, score in Qf_eval_results.items():
                    metrics_str += f" {metric}: {score:.4f} |"
                print(metrics_str)
                print('\n')
                
            except Exception as e:
                print(f"Error computing NLP metrics: {e}")

        # Add sample statistics to metrics
        val_metrics.update({
            'valid_samples': valid_samples,
            'empty_samples': empty_samples,
            'total_samples': len(Qf_predictions)
        })
        
        return val_metrics
    
    def compute_scores(self, gts, res):
        """
        Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

        :param gts: Dictionary with the image ids and their gold captions,
        :param res: Dictionary with the image ids ant their generated captions
        :print: Evaluation score (the mean of the scores of all the instances) for each measure
        """

        # Set up scorers
        scorers = self.get_scorer()
        eval_res = {}
        # Compute score for each metric
        for scorer, method in scorers:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res
    
    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Validation:"
        print_freq = 10
        model.eval()
        
        all_predictions = []
        all_ground_truths = []
        all_study_id = []
        
        with torch.no_grad():
            for batch_idx, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                
                
                outputs = self.valid_step(model=model, samples=samples)
                
                all_predictions.extend(outputs['predicted_captions'])
                all_ground_truths.extend(outputs['gt_captions'])
                all_study_id.extend(outputs['study_id'])
        if is_dist_avail_and_initialized():
            world_size = dist.get_world_size()
            all_predictions_list = [None] * world_size
            all_ground_truths_list = [None] * world_size
            all_study_id_list = [None] * world_size

            dist.all_gather_object(all_predictions_list, all_predictions)
            dist.all_gather_object(all_ground_truths_list, all_ground_truths)
            dist.all_gather_object(all_study_id_list, all_study_id)
            # dist.barrier()
            if is_main_process():
                all_predictions = []
                all_ground_truths = []
                all_study_id = []
                for pred_list, gt_list, study_id_list in zip(all_predictions_list, all_ground_truths_list, all_study_id_list):
                    all_predictions.extend(pred_list)
                    all_ground_truths.extend(gt_list)
                    all_study_id.extend(study_id_list)
            
                study_id_indices = {
                    study_id: [i for i, sid in enumerate(all_study_id) if sid == study_id]
                    for study_id in set(all_study_id)
                }
                
                duplicates = {
                    study_id: indices 
                    for study_id, indices in study_id_indices.items() 
                    if len(indices) > 1
                }
                
                if duplicates:
                    print("\n=== Duplicate Study IDs ===")
                    for study_id, indices in duplicates.items():
                        print(f"\nStudy ID {study_id} appears at indices: {indices}")
                        print("First occurrence:")
                        print(f"Pred: {all_predictions[indices[0]][:100]}...")
                        print(f"GT: {all_ground_truths[indices[0]][:100]}...")
                        print("Second occurrence:")
                        print(f"Pred: {all_predictions[indices[1]][:100]}...")
                        print(f"GT: {all_ground_truths[indices[1]][:100]}...")
                        
                unique_indices = list({study_id: idx for idx, study_id in enumerate(all_study_id)}.values())
                all_predictions = [all_predictions[i] for i in unique_indices]
                all_ground_truths = [all_ground_truths[i] for i in unique_indices]
                all_study_id = [all_study_id[i] for i in unique_indices]
                print(f"total number of unique_indices: {len(unique_indices)}")
                print(f"Total samples processed: {len(all_predictions)}")
        
        metrics = {}
        
        # nlp metrics
        caption_metrics = self.compute_metrics(all_predictions, all_ground_truths)
        metrics['caption'] = caption_metrics
        
        # ce metrics
        clinical_metrics = self.compute_clinical_metrics(all_predictions, all_ground_truths, all_study_id)
        metrics['clinical'] = clinical_metrics
        
        if is_main_process():
            # 
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = f"metrics_{timestamp}.json"
            
            print(f"\nSaving metrics to {metrics_file}")
            try:
                #
                formatted_metrics = {
                    'timestamp': timestamp,
                    'caption_metrics': {
                        k: float(f"{v:.4f}") if isinstance(v, (float, np.float32, np.float64)) else v
                        for k, v in caption_metrics.items()
                    },
                    'clinical_metrics': {
                        k: float(f"{v:.4f}") if isinstance(v, (float, np.float32, np.float64)) else v
                        for k, v in clinical_metrics.items()
                    }
                }
                
                # 
                if len(all_predictions) > 0:
                    examples = [
                        {'pred': pred, 'true': true}
                        for pred, true in zip(all_predictions[-2:], all_ground_truths[-2:])
                    ]
                    formatted_metrics['examples'] = examples
                
                # 
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(formatted_metrics, f, indent=4, ensure_ascii=False)
                
                # 
                print("\n" + "="*50 + " Clinical Metrics " + "="*50)
                if clinical_metrics:
                    for metric, value in clinical_metrics.items():
                        if isinstance(value, (float, np.float32, np.float64)):
                            print(f"{metric}: {value:.4f}")
                        else:
                            print(f"{metric}: {value}")
                else:
                    print("No clinical metrics available")
                print("="*110)
                
                # 
                print("\n" + "="*50 + " Caption Metrics " + "="*50)
                if caption_metrics:
                    for metric, value in caption_metrics.items():
                        if isinstance(value, (float, np.float32, np.float64)):
                            print(f"{metric}: {value:.4f}")
                        else:
                            print(f"{metric}: {value}")
                else:
                    print("No caption metrics available")
                print("="*110)
                
                if len(all_predictions) > 0:
                    print("\n" + "="*50 + " Examples " + "="*50)
                    for pred, true in zip(all_predictions[-2:], all_ground_truths[-2:]):
                        print(f"Pred: {pred}")
                        print(f"True: {true}")
                        print("-" * 100)
                
                print(f"\nMetrics saved to: {os.path.abspath(metrics_file)}")
                
            except Exception as e:
                print(f"Error saving metrics: {e}")
                traceback.print_exc()
        
        if is_dist_avail_and_initialized():
            dist.barrier()
        
        return metrics
    
    def create_analysis_table(self, pred_reports, gt_reports, all_study_ids, pred_labels, gt_labels):
        """
        Creates a comprehensive analysis table using existing CheXpert labels
        
        Args:
            pred_reports: list of predicted reports
            gt_reports: list of ground truth reports
            all_study_ids: list of study IDs
            pred_labels: numpy array of prediction labels from CheXpert
            gt_labels: numpy array of ground truth labels from CheXpert
        
        Returns:
            DataFrame with detailed analysis
        """
        try:
            # Create the base DataFrame
            analysis_df = pd.DataFrame({
                'study_id': all_study_ids,
                'predicted_report': pred_reports,
                'ground_truth_report': gt_reports
            })
            
            # Add disease labels columns with prefixes
            for i, category in enumerate(CATEGORIES):
                # Get labels for this category
                category_preds = pred_labels[:, i].copy()
                category_truths = gt_labels[:, i].copy()
                
                # Convert NaN and -1 to 0 (consistent with compute_clinical_metrics)
                category_preds[np.isnan(category_preds)] = 0
                category_preds[category_preds == -1] = 0
                category_truths[np.isnan(category_truths)] = 0
                category_truths[category_truths == -1] = 0
                
                # Add to DataFrame
                analysis_df[f'pred_{category}'] = category_preds
                analysis_df[f'gt_{category}'] = category_truths
                analysis_df[f'{category}_match'] = (category_preds == category_truths)
            
            # Add summary columns
            analysis_df['total_matches'] = sum(
                analysis_df[f'{category}_match'] 
                for category in CATEGORIES
            )
            analysis_df['match_percentage'] = (
                analysis_df['total_matches'] / len(CATEGORIES) * 100
            ).round(2)
            
            # Save the analysis
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"report_analysis_{timestamp}.csv"
            analysis_df.to_csv(output_file, index=False)
            print(f"\nDetailed analysis saved to: {output_file}")
            
            return analysis_df
            
        except Exception as e:
            print(f"Error in creating analysis table: {e}")
            traceback.print_exc()
            return None

