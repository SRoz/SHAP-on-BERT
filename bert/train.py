import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm, trange
#from tensorboardX import SummaryWriter

#from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertModel, BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer

import random
import os
import pickle
import numpy as np
import logging
logger = logging.getLogger()

def train(processor, parameters):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, False, parameters['FP16']))


    TRAIN_BATCH_SIZE_ = parameters['TRAIN_BATCH_SIZE'] // parameters['GRADIENT_ACCUMULATION_STEPS']

    random.seed(parameters['SEED'])
    np.random.seed(parameters['SEED'])
    torch.manual_seed(parameters['SEED'])
    if n_gpu > 0:
        torch.cuda.manual_seed_all(parameters['SEED'])


    if os.path.exists(parameters['OUTPUT_DIR']) and os.listdir(parameters['OUTPUT_DIR']) and parameters['DO_TRAIN']:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(parameters['OUTPUT_DIR']))

    if not os.path.exists(parameters['OUTPUT_DIR']):
        os.makedirs(parameters['OUTPUT_DIR'])

    task_name = parameters['TASK_NAME'].lower()

    output_mode = 'classification'

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(parameters['BERT_MODEL'], do_lower_case=parameters['DO_LOWER_CASE'])

    train_examples = None
    num_train_optimization_steps = None
    if parameters['DO_TRAIN']:
        train_examples = processor.get_train_examples()
        num_train_optimization_steps = int(
            len(train_examples) / TRAIN_BATCH_SIZE_ / parameters['GRADIENT_ACCUMULATION_STEPS']) * parameters['NUM_TRAIN_EPOCHS']

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(parameters['BERT_MODEL'],
                cache_dir=parameters['CACHE_DIR'],
                num_labels=num_labels)

    model.to(device)
    model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if parameters['DO_TRAIN']:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=parameters['LEARNING_RATE'],
                                warmup=parameters['WARMUP_PROPORTION'],
                                t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if parameters['DO_TRAIN']:
        train_features = convert_examples_to_features(
            train_examples, label_list, parameters['MAX_SEQ_LENGTH'], tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", TRAIN_BATCH_SIZE_)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if parameters['LOCAL_RANK'] == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE_)

        model.train()
        for _ in trange(int(parameters['NUM_TRAIN_EPOCHS']), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if parameters['GRADIENT_ACCUMULATION_STEPS'] > 1:
                    loss = loss / parameters['GRADIENT_ACCUMULATION_STEPS']

                if parameters['FP16']:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % parameters['GRADIENT_ACCUMULATION_STEPS'] == 0:
                    if parameters['FP16']:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = parameters['LEARNING_RATE'] * warmup_linear.get_lr(global_step, parameters['WARMUP_PROPORTION'])
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(parameters['OUTPUT_DIR'], parameters['WEIGHTS_NAME'])
        output_config_file = os.path.join(parameters['OUTPUT_DIR'], parameters['CONFIG_NAME'])

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(parameters['OUTPUT_DIR'])

        logging.info(tr_loss)
        
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        logging.info("Getting from pretrained")
        model = BertForSequenceClassification.from_pretrained(parameters['OUTPUT_DIR'], num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(parameters['OUTPUT_DIR'], do_lower_case=parameters['DO_LOWER_CASE'])

    model.to(device);
    return model, tokenizer


def eval(model, processor, parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_mode = 'classification'
    task_name = parameters['TASK_NAME'].lower()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    eval_examples = processor.get_dev_examples()
    cached_eval_features_file = os.path.join(parameters['DATA_DIR'], 'dev_{0}_{1}_{2}'.format(
        list(filter(None, parameters['BERT_MODEL'].split('/'))).pop(),
                    str(parameters['MAX_SEQ_LENGTH']),
                    str(task_name)))
    try:
        with open(cached_eval_features_file, "rb") as reader:
            eval_features = pickle.load(reader)
    except:
        eval_features = convert_examples_to_features(
            eval_examples, label_list, parameters['MAX_SEQ_LENGTH'], tokenizer, output_mode)
        if parameters['LOCAL_RANK'] == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving eval features into cached file %s", cached_eval_features_file)
            with open(cached_eval_features_file, "wb") as writer:
                pickle.dump(eval_features, writer)


    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", parameters['EVAL_BATCH_SIZE'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    if parameters['LOCAL_RANK'] == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)  # Note that this sampler samples randomly
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=parameters['EVAL_BATCH_SIZE'])

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    out_label_ids = None

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()
    def compute_metrics(preds, labels):
        return {"acc": simple_accuracy(preds, labels)}

    result = compute_metrics(preds, out_label_ids)

    loss = None # tr_loss/global_step if parameters['DO_TRAIN'] else None

    result['eval_loss'] = eval_loss
    #result['global_step'] = global_step
    #result['loss'] = loss


    output_eval_file = os.path.join(parameters['OUTPUT_DIR'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    