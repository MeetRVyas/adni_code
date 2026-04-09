import argparse
import json

from module import Cross_Validator, Logger

def run_batch(models, classifier_map):
    logger = Logger("batch_" + str(hash(str(models)))[:8])
    logger.info(f"Starting validation for {models}")
    logger.info(f"Classifier mapping: {classifier_map}")
    
    validator = Cross_Validator(
        models,
        logger,
        model_classifier_map=classifier_map
    )
    validator.run()
    logger.info("Validation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--classifier_map", type=str, required=True)

    args = parser.parse_args()

    models = json.loads(args.models)
    classifier_map = json.loads(args.classifier_map)

    run_batch(models, classifier_map)