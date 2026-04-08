from module.cross_validation import Cross_Validator
from module.utils import Logger

def run_batch():
    models = ["resnet18"]
    classifier_map = {"resnet18" : ['baseline', 'progressive', 'evidential', 'metric_learning', 'regularized',
                 'attention_enhanced', 'progressive_evidential', 'clinical_grade', 'hybrid_transformer', 'ultimate']}

    logger = Logger("batch_" + str(hash(str(models)))[:8])
    logger.info(f"Starting validation for {models}")
    logger.info(f"Classifier mapping: {classifier_map}")
    
    try:
        validator = Cross_Validator(
            models,
            logger,
            model_classifier_map=classifier_map
        )
        validator.run()
        logger.info("Validation complete")
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise

if __name__ == "__main__":
    run_batch()