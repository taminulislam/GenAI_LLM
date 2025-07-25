"""
Comprehensive Medical LLM Configuration
Supports multiple datasets, models, and advanced evaluation
"""

class ComprehensiveConfig:
    """Comprehensive configuration for medical LLM training"""
    
    def __init__(self):
        # Dataset configurations
        self.dataset_configs = {
            'medmcqa': {
                'name': 'medmcqa',
                'train_samples': 5000,
                'eval_samples': 500,
                'test_samples': 500
            },
            'pubmedqa': {
                'name': 'pubmed_qa',
                'subset': 'pqa_labeled',
                'train_samples': 3000,
                'eval_samples': 600,
                'test_samples': 500
            },
            'medical_flashcards': {
                'name': 'medalpaca/medical_meadow_medical_flashcards',
                'train_samples': 2000,
                'eval_samples': 400,
                'test_samples': 300
            }
        }
        
        # Model configurations
        self.model_configs = {
            'dialogpt_small': {
                'model_name': 'microsoft/DialoGPT-small',
                'description': 'Fast, efficient model'
            },
            'dialogpt_medium': {
                'model_name': 'microsoft/DialoGPT-medium', 
                'description': 'Balanced performance'
            },
            'gpt2': {
                'model_name': 'gpt2',
                'description': 'Standard GPT-2 model'
            },
            'distilgpt2': {
                'model_name': 'distilgpt2',
                'description': 'Lightweight GPT-2'
            }
        }
        
        # Training configuration
        self.training_config = {
            'num_train_epochs': 5,
            'per_device_train_batch_size': 4,
            'learning_rate': 2e-4,
            'warmup_steps': 100,
            'weight_decay': 0.01,
            'logging_steps': 50,
            'eval_steps': 200,
            'save_steps': 500
        }
        
        # LoRA configuration
        self.lora_config = {
            'r': 32,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'target_modules': ['c_attn', 'c_proj', 'c_fc']
        }
        
        # Evaluation configuration
        self.eval_config = {
            'max_eval_samples': 100,
            'max_new_tokens': 50,
            'temperature': 0.7,
            'do_sample': True
        }
        
        # Directories
        self.models_dir = "./models"
        self.data_dir = "./data"
        self.experiments_dir = "./experiments" 