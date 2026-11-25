import argparse
import torch
import torch.backends.cudnn as cudnn
import beitscore.utils as utils
from timm.models import create_model
from beitscore.data.datasets import create_downstream_dataset
from beitscore.engine import RegressorHandler, evaluate, train
from beitscore.pp.handler import QuantHandler
from beitscore.stage1.handler import BCEHandler
from beitscore.models.checkpoint import load_model_and_may_interpolate_beitscore, load_model_and_may_interpolate_beitscorepp
from beitscore.optim_factory import create_optimizer, LayerDecayValueAssigner, get_is_head_flag_for_vit
from beitscore.utils import NativeScalerWithGradNormCount as NativeScaler
import beitscore.models  # noqa: F401  (registers the model configs)
import logging
import os
from beitscore.paths import DATA_ROOT

# Defaults that differ between the two metrics. BEiTScore (stage1, regression)
# keeps the settings its released runs used; BEiTScore++ (quant) uses the paper's.
TASK_DEFAULTS = {
    "stage1": dict(early_stopping=5, use_clusters=False, chained_construction=False),
    "regression": dict(early_stopping=5, use_clusters=False, chained_construction=False),
    "quant": dict(early_stopping=15, use_clusters=True, chained_construction=True),
}

def get_args():
    parser = argparse.ArgumentParser('Train BEiTScore (stage1, regression) or BEiTScore++ (quant)')

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int, help='Seed for torch, numpy and random, and for the data shuffles')
    
    # Model parameters
    parser.add_argument('--model_size', default='base', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--model_type', type=str, choices=['none', 'indomain', 'nlvr2'], default="indomain", required=False, help='Select the type of checkpoint')
    parser.add_argument('--quant_ci', type=int, default=75, help='Select the target ci (e.g. ci = 90 -> [0.05, 0.095])')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--checkpoint_activations', action='store_true', default=None, help='Enable checkpointing to save your memory.')
    parser.add_argument('--input_size', default=480, type=int, help='images input size')
    parser.add_argument('--sentencepiece_model', default='beit3.spm', type=str, help='Sentencepiece model path for the pretrained model.')
    parser.add_argument('--num_max_bpe_tokens', type=int, default=256)

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    parser.add_argument('--drop_path', type=float, default=0.15, metavar='PCT', help='Drop path rate (default: 0.15)')
    parser.add_argument('--dropout', type=float, default=0, metavar='PCT', help='Dropout rate (default: 0)')
    parser.add_argument('--attention_dropout', type=float, default=0, metavar='PCT', help='Attention dropout rate (default: 0)')
    parser.add_argument('--activation_dropout', type=float, default=0, metavar='PCT', help='Activation dropout rate (default: 0)')
    parser.add_argument('--length_penalty', type=float, default=0.6)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.add_argument('--init_scale', type=float, default=0.02, help='Initial scale for weight initialization')
    
    #Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.98], type=float, nargs='+', metavar='BETA', help='Optimizer betas')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--instances_per_eval', default=-1, type=int, help='if > 0 run evaluation every N instances, else use epochs')
    parser.add_argument('--layer_decay', type=float, default=1)
    parser.add_argument('--task_head_lr_weight', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
    parser.add_argument('--lr', type=float, default=5e-7, metavar='LR', help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, metavar='LR', help='Lower bound of the learning rate for schedulers that reach 0')
    parser.add_argument('--warmup_lr', type=float, default=1e-7, metavar='LR', help='Learning rate at the start of the warmup')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    
    
    #Finetuning parameters
    parser.add_argument('--get_checkpoint_from_finetuning', action='store_true',help='Needs to be active to get checkpoint based on task, select which task to be resumed and run')
    parser.add_argument('--task_resume', type=str, default="", choices=['','stage1', 'regression'], help='Name of checkpoint task to start fine-tuning')
    parser.add_argument('--use_run', default=0, type=int, help='Referred to the finetuned run')
    parser.add_argument('--checkpoint_number', default=-1, type=int, help='Referred to the checkpoint number, the default -1 represents the best performing checkpoint')
    parser.add_argument('--init_checkpoint', default=None, type=str, help='Explicit checkpoint to start from (e.g. a released BEiTScore model for --benchmark quant); overrides the options above')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--early_stopping', default=None, type=int, help='Evaluations without improvement before training stops (default: 15 for quant, 5 otherwise)')
    
    #Dataset Parameters
    parser.add_argument('--data_path', default=DATA_ROOT, type=str,help='Data directory')
    parser.add_argument('--benchmark', type=str, choices=['stage1', 'regression', 'quant'], default="stage1", required=False, help='Select the benchmark')
    parser.add_argument('--bias', type=float, default=-2.5,
                        help='Initial offset of the lower and upper quantile heads (BEiTScore++). '
                             'It is probit-scaled: the outermost quantile gets the full offset, '
                             'with opposite signs on the two sides (see models/checkpoint.py)')
    parser.add_argument('--median_weight', '--coef', dest='median_weight', type=float, default=1.0,
                        help='Weight of the L1 term on the median in the BEiTScore++ loss')
    parser.add_argument('--use_clusters', action=argparse.BooleanOptionalAction, default=None,
                        help='Batches drawn from one image cluster at a time (default: on for quant, off otherwise)')
    parser.add_argument('--pinball', action='store_true', help='BEiTScore++: pinball loss instead of the centered interval loss')
    parser.add_argument('--chained_construction', action=argparse.BooleanOptionalAction, default=None,
                        help='Monotone chained quantile head (default: on for quant)')
    parser.add_argument('--stage1_original_foil', action='store_true',
                        help='Stage 1: also draw the human-written incorrect captions of FOIL-it (off for the released models)')
    parser.add_argument('--quantiles',  default=[0.9,0.85,0.8,0.75,0.7], type=float, nargs='+', help='Select the quantiles for the quantile regression')
    parser.set_defaults(pin_mem=True)
    args = parser.parse_args()
    for key, value in TASK_DEFAULTS[args.benchmark].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args

def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Add handler only if not already added
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

def main(args):
    device = torch.device(args.device)
    logger = setup_logger()
    utils.set_seed(args.seed)

    #sets up logs/directories
    cudnn.benchmark = True
    
    args.task = args.benchmark
    utils.update_log_and_save_dir(args)



    if args.task == "stage1":
        args.greater_is_better = True
        args.task_key = "f1"
    elif args.task == "regression":
        args.greater_is_better = True
        args.task_key = "tau_c"
    elif args.task == "quant":
        args.benchmark = "regression"
        args.greater_is_better = True
        args.task_key = "ups_90.0"
    
    if args.pinball:
        criterion = "pinball"
    else:
        criterion = "CI"

    args.model = "beit3_{}_patch16".format(args.model_size)
    args.model = "{}_{}".format(args.model, args.input_size)
    
    model_config = "%s_%s" % (args.model, args.task)
    model = create_model(model_config, pretrained=False, dropout=args.dropout, drop_path_rate=args.drop_path, vocab_size=args.vocab_size, checkpoint_activations=args.checkpoint_activations, chained_construction=args.chained_construction,quantiles=args.quantiles,)
        
    checkpoint = args.init_checkpoint or utils.get_resume_and_finetune_checkpoint(args)
    args.sentencepiece_model = "{}/{}".format(args.models_dir, args.sentencepiece_model)

    logger.info("Start Interpolate")
    if args.task == "quant":
        load_model_and_may_interpolate_beitscorepp(args,checkpoint, model, args.model_key, args.model_prefix)
    else:
        load_model_and_may_interpolate_beitscore(checkpoint, model, args.model_key, args.model_prefix)
    logger.info("Send Model to GPU")
    model.to(device)

    zip_loader = create_downstream_dataset(args, is_eval=False)
    data_loader_test = create_downstream_dataset(args, is_eval=True)
    data_loader_train = zip_loader["data_loader_train"]
    data_loader_val = zip_loader["data_loader_val"]
    logger.info("Train Dataset Length: {}".format(len(data_loader_train.dataset)))
    logger.info("Val Dataset Length: {}".format(len(data_loader_val.dataset)))
    logger.info("Test Dataset Length: {}".format(len(data_loader_test.dataset)))
    
    model.max_value = 1
    model.min_value = 0
    
    if args.task == "stage1":
        task_handler_train = BCEHandler()
        task_handler_val = BCEHandler()
        task_handler_test = BCEHandler()
    elif args.task == "regression":
        task_handler_train = RegressorHandler()
        task_handler_val = RegressorHandler()
        task_handler_test = RegressorHandler()
    elif args.task == "quant":
        assert (args.quant_ci > 0) and (args.quant_ci < 100) 
        task_handler_train = QuantHandler(CI = model.quantiles, coef=args.median_weight, criterion = criterion)
        task_handler_val = QuantHandler(CI = model.quantiles, coef=args.median_weight, criterion = criterion)
        task_handler_test = QuantHandler(CI = model.quantiles, coef=args.median_weight, criterion = criterion)
    

    num_layers = model.get_num_layers()
    if args.layer_decay < 1.0:
        lrs = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner(lrs)
    elif args.task_head_lr_weight > 1:
        assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
    else:
        assigner = None

    if assigner is not None:
        logger.info("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    
    optimizer = create_optimizer(
        args, skip_list=skip_weight_decay_list, model=model,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()
    
    utils.auto_load_resume_model(
        args=args, resume=None, model_without_ddp=model,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=None)
    
    logger.info("Start Training")
    train(args, model, optimizer, loss_scaler, task_handler_train, task_handler_val, data_loader_train=data_loader_train, data_loader_val=data_loader_val, log_writer=None, model_ema=None, device=device)
    
    # The test evaluation uses the checkpoint selected on the validation split.
    best = os.path.join(args.output_dir, "checkpoint-best.pth")
    if os.path.exists(best):
        logger.info("Reloading %s for the test evaluation" % best)
        state = torch.load(best, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(state)
    evaluate(data_loader_test, model, device, task_handler_test, save_values=True)

if __name__ == "__main__":
    opts = get_args()
    main(opts)
