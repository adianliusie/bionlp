import logging
import pickle
import os

from src.handlers.trainer import Trainer
from src.handlers.evaluater import Evaluater

from src.utils.general import save_json, load_json
from src.utils.parser import get_model_parser, get_train_parser

# Load logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    model_parser = get_model_parser()
    train_parser = get_train_parser()

    # Parse system input arguments
    model_args, moargs = model_parser.parse_known_args()
    train_args, toargs = train_parser.parse_known_args()

    # Making sure no unkown arguments are given
    assert set(moargs).isdisjoint(toargs), f"{set(moargs) & set(toargs)}"

    logger.info(model_args.__dict__)
    logger.info(train_args.__dict__)
    
    preds = {}
    for fold in [0, 1, 2, 3, 4]:
        # train model in cross validation set up
        dataset = f"bionlp-{fold}"
        train_args.dataset = dataset
        path = f"{model_args.path}/fold-{fold}"
        trainer = Trainer(path, model_args)
        trainer.train(train_args)
        
        # run evaluation on fold 
        evaluater = Evaluater(path, train_args.device)
        pred_texts = evaluater.load_pred_texts(dataset, 'test')
        label_texts = evaluater.load_label_texts(dataset, 'test')
        evaluater.calculate_rouge(pred_texts, label_texts, display=True)
        evaluater.calculate_bleu(pred_texts, label_texts, display=True)
        
        # add results to overall outputs
        preds = {**preds, **pred_texts}
    
    # save overall outputs 
    eval_path = os.path.join(model_args.path, 'eval')
    if not os.path.isdir(eval_path):
        os.mkdir(eval_path)

    save_path = os.path.join(eval_path, 'bionlp_test.pk')
    with open(save_path, 'wb') as handle:
        pickle.dump(preds, handle)
