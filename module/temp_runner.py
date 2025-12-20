
import sys
from .cross_validation import Cross_Validator
from .utils import Logger

def run_batch():
    print("--- Subprocess Started ---")

    # Placeholders replaced by the main script
    models = ['vit_base_patch16_224.augreg2_in21k_ft_in1k', 'vit_base_patch16_224', 'vit_tiny_patch16_224.augreg_in21k_ft_in1k']
    use_aug = False

    # logic
    logger = Logger("05_12_2025")
    logger.info(f"Starting Validation for {models}")

    validator = Cross_Validator(models, logger, use_aug = use_aug)
    validator.run()

    logger.info("Validation Complete")

if __name__ == "__main__":
    run_batch()
