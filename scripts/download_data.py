#!/usr/bin/env python3
"""
Medical Dataset Download and Preparation Script
Downloads and prepares medical datasets for LLM fine-tuning
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import MedicalDataLoader, load_medical_data, preprocess_medical_data
from src.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDatasetDownloader:
    """Handles downloading and preparing medical datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.downloaded_datasets = {}
        
    def download_medical_qa_datasets(self) -> Dict[str, str]:
        """Download medical Q&A datasets"""
        logger.info("📥 Downloading Medical Q&A Datasets...")
        
        datasets_info = {}
        
        try:
            # Dataset 1: lavita/medical-qa-datasets
            logger.info("Downloading lavita/medical-qa-datasets...")
            from datasets import load_dataset
            
            dataset = load_dataset("lavita/medical-qa-datasets", "all-processed", split="train[:5000]")
            save_path = self.data_dir / "medical_qa_lavita"
            dataset.save_to_disk(str(save_path))
            datasets_info["medical_qa_lavita"] = {
                "path": str(save_path),
                "size": len(dataset),
                "description": "General medical Q&A dataset"
            }
            logger.info(f"✅ Saved {len(dataset)} samples to {save_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not download lavita/medical-qa-datasets: {e}")
        
        try:
            # Dataset 2: BI55/MedText
            logger.info("Downloading BI55/MedText...")
            dataset = load_dataset("BI55/MedText", split="train")
            save_path = self.data_dir / "medtext_bi55"
            dataset.save_to_disk(str(save_path))
            datasets_info["medtext_bi55"] = {
                "path": str(save_path),
                "size": len(dataset),
                "description": "Medical text scenarios"
            }
            logger.info(f"✅ Saved {len(dataset)} samples to {save_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not download BI55/MedText: {e}")
        
        try:
            # Dataset 3: Medical dialog dataset
            logger.info("Downloading medical dialog dataset...")
            dataset = load_dataset("medical_dialog", "processed.en", split="train[:2000]")
            save_path = self.data_dir / "medical_dialog"
            dataset.save_to_disk(str(save_path))
            datasets_info["medical_dialog"] = {
                "path": str(save_path),
                "size": len(dataset),
                "description": "Medical consultation dialogs"
            }
            logger.info(f"✅ Saved {len(dataset)} samples to {save_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not download medical_dialog: {e}")
        
        return datasets_info
    
    def download_benchmark_datasets(self) -> Dict[str, str]:
        """Download medical benchmark datasets for evaluation"""
        logger.info("📥 Downloading Medical Benchmark Datasets...")
        
        datasets_info = {}
        
        try:
            # MedMCQA dataset
            logger.info("Downloading MedMCQA dataset...")
            from datasets import load_dataset
            
            dataset = load_dataset("openlifescienceai/medmcqa", split="validation[:1000]")
            save_path = self.data_dir / "medmcqa_benchmark"
            dataset.save_to_disk(str(save_path))
            datasets_info["medmcqa"] = {
                "path": str(save_path),
                "size": len(dataset),
                "description": "Medical multiple choice questions"
            }
            logger.info(f"✅ Saved {len(dataset)} samples to {save_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not download MedMCQA: {e}")
        
        try:
            # PubMedQA dataset
            logger.info("Downloading PubMedQA dataset...")
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="test[:500]")
            save_path = self.data_dir / "pubmedqa_benchmark"
            dataset.save_to_disk(str(save_path))
            datasets_info["pubmedqa"] = {
                "path": str(save_path),
                "size": len(dataset),
                "description": "PubMed question answering"
            }
            logger.info(f"✅ Saved {len(dataset)} samples to {save_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not download PubMedQA: {e}")
        
        return datasets_info
    
    def create_dummy_datasets(self) -> Dict[str, str]:
        """Create dummy medical datasets for testing"""
        logger.info("🧪 Creating Dummy Medical Datasets...")
        
        from datasets import Dataset
        
        # Dummy training data
        dummy_training_data = [
            {
                "question": "What is the normal human body temperature?",
                "answer": "The normal human body temperature is approximately 98.6°F (37°C), though it can range from 97°F to 99°F.",
                "context": "Body temperature regulation"
            },
            {
                "question": "What causes Type 1 diabetes?",
                "answer": "Type 1 diabetes is caused by the immune system destroying insulin-producing beta cells in the pancreas.",
                "context": "Endocrine disorders"
            },
            {
                "question": "What are the symptoms of hypertension?",
                "answer": "Hypertension often has no symptoms but can cause headaches, shortness of breath, and nosebleeds in severe cases.",
                "context": "Cardiovascular health"
            },
            {
                "question": "How is pneumonia diagnosed?",
                "answer": "Pneumonia is diagnosed through chest X-rays, blood tests, and physical examination of breathing sounds.",
                "context": "Respiratory diseases"
            },
            {
                "question": "What is the function of red blood cells?",
                "answer": "Red blood cells carry oxygen from the lungs to body tissues and transport carbon dioxide back to the lungs.",
                "context": "Hematology"
            }
        ] * 50  # Repeat to create 250 samples
        
        dummy_training_dataset = Dataset.from_list(dummy_training_data)
        training_save_path = self.data_dir / "dummy_medical_training"
        dummy_training_dataset.save_to_disk(str(training_save_path))
        
        # Dummy evaluation data
        dummy_eval_data = [
            {
                "question": "What is the primary function of the heart?",
                "choices": ["A) Digestion", "B) Circulation", "C) Respiration", "D) Excretion"],
                "answer": "B",
                "context": "The heart pumps blood throughout the body."
            },
            {
                "question": "Which vitamin deficiency causes scurvy?",
                "choices": ["A) Vitamin A", "B) Vitamin B", "C) Vitamin C", "D) Vitamin D"],
                "answer": "C",
                "context": "Vitamin C is essential for collagen synthesis."
            },
            {
                "question": "What is the normal range for adult resting heart rate?",
                "choices": ["A) 40-60 bpm", "B) 60-100 bpm", "C) 100-120 bpm", "D) 120-140 bpm"],
                "answer": "B",
                "context": "Normal resting heart rate for adults."
            }
        ] * 20  # Create 60 evaluation samples
        
        dummy_eval_dataset = Dataset.from_list(dummy_eval_data)
        eval_save_path = self.data_dir / "dummy_medical_evaluation"
        dummy_eval_dataset.save_to_disk(str(eval_save_path))
        
        datasets_info = {
            "dummy_training": {
                "path": str(training_save_path),
                "size": len(dummy_training_dataset),
                "description": "Dummy medical training data"
            },
            "dummy_evaluation": {
                "path": str(eval_save_path),
                "size": len(dummy_eval_dataset),
                "description": "Dummy medical evaluation data"
            }
        }
        
        logger.info(f"✅ Created dummy training dataset: {len(dummy_training_dataset)} samples")
        logger.info(f"✅ Created dummy evaluation dataset: {len(dummy_eval_dataset)} samples")
        
        return datasets_info
    
    def preprocess_datasets(self, datasets_info: Dict[str, str]):
        """Preprocess downloaded datasets"""
        logger.info("🔄 Preprocessing Datasets...")
        
        data_loader = MedicalDataLoader()
        
        for dataset_name, info in datasets_info.items():
            try:
                logger.info(f"Processing {dataset_name}...")
                
                # Load the saved dataset
                from datasets import load_from_disk
                dataset = load_from_disk(info["path"])
                
                # Set up data loader with this dataset
                data_loader.dataset = dataset
                data_loader.preprocess_dataset()
                
                # Save preprocessed dataset
                processed_path = Path(info["path"]).parent / f"{dataset_name}_processed"
                data_loader.save_processed_dataset(str(processed_path))
                
                # Update info
                datasets_info[dataset_name]["processed_path"] = str(processed_path)
                datasets_info[dataset_name]["processed_size"] = len(data_loader.processed_dataset)
                
                logger.info(f"✅ Processed {dataset_name}: {len(data_loader.processed_dataset)} samples")
                
            except Exception as e:
                logger.error(f"❌ Error processing {dataset_name}: {e}")
    
    def save_dataset_info(self, datasets_info: Dict[str, str]):
        """Save dataset information to file"""
        import json
        
        info_file = self.data_dir / "datasets_info.json"
        with open(info_file, 'w') as f:
            json.dump(datasets_info, f, indent=2)
        
        logger.info(f"💾 Dataset information saved to {info_file}")
    
    def download_all(self, include_benchmarks: bool = True, create_dummy: bool = True) -> Dict[str, str]:
        """Download all available datasets"""
        logger.info("🚀 Starting Medical Dataset Download Process...")
        
        all_datasets = {}
        
        # Download training datasets
        training_datasets = self.download_medical_qa_datasets()
        all_datasets.update(training_datasets)
        
        # Download benchmark datasets
        if include_benchmarks:
            benchmark_datasets = self.download_benchmark_datasets()
            all_datasets.update(benchmark_datasets)
        
        # Create dummy datasets if requested or if no real datasets were downloaded
        if create_dummy or not all_datasets:
            dummy_datasets = self.create_dummy_datasets()
            all_datasets.update(dummy_datasets)
        
        # Preprocess all datasets
        if all_datasets:
            self.preprocess_datasets(all_datasets)
            self.save_dataset_info(all_datasets)
        
        return all_datasets

def print_dataset_summary(datasets_info: Dict[str, str]):
    """Print a summary of downloaded datasets"""
    logger.info("📊 Dataset Download Summary:")
    logger.info("=" * 60)
    
    total_samples = 0
    for name, info in datasets_info.items():
        size = info.get("size", 0)
        description = info.get("description", "No description")
        total_samples += size
        logger.info(f"{name:20} | {size:6} samples | {description}")
    
    logger.info("=" * 60)
    logger.info(f"Total samples downloaded: {total_samples}")
    
    if datasets_info:
        logger.info("\n✅ Datasets ready for training!")
        logger.info("Next step: Run 'python scripts/train_model.py' to start training")
    else:
        logger.warning("⚠️ No datasets were downloaded successfully")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download medical datasets for LLM training")
    parser.add_argument("--data-dir", default="data", help="Directory to store datasets")
    parser.add_argument("--no-benchmarks", action="store_true", help="Skip downloading benchmark datasets")
    parser.add_argument("--no-dummy", action="store_true", help="Skip creating dummy datasets")
    parser.add_argument("--dummy-only", action="store_true", help="Only create dummy datasets (for testing)")
    
    args = parser.parse_args()
    
    logger.info("🏥 Medical Dataset Downloader")
    logger.info("=" * 50)
    
    downloader = MedicalDatasetDownloader(args.data_dir)
    
    if args.dummy_only:
        logger.info("🧪 Creating dummy datasets only...")
        datasets_info = downloader.create_dummy_datasets()
        downloader.preprocess_datasets(datasets_info)
        downloader.save_dataset_info(datasets_info)
    else:
        datasets_info = downloader.download_all(
            include_benchmarks=not args.no_benchmarks,
            create_dummy=not args.no_dummy
        )
    
    print_dataset_summary(datasets_info)

if __name__ == "__main__":
    main() 