import os
from random import sample
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchtext import datasets

import logging
logger = logging.getLogger()

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
class IMDBProcessor():
    """Processor for the DA commentary fields for the regression task"""
    
    def __init__(self, device=0, data_path="./"):
        self.train_iter, self.test_iter = datasets.IMDB.iters(batch_size=1, 
                                                    root=data_path,vectors=None)
        
    def get_train_examples(self):
        """See base class."""
        return self._create_examples( "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples( "test")

    def get_labels(self):
        """See base class."""
        return ["pos","neg"]
    
    def _create_examples(self, set_type):
        """Creates examples for the training and dev sets."""
        
        assert set_type in ('train', 'test')
        
        iter_set = self.train_iter if set_type=='train' else self.test_iter
        
        examples = []
        for (i, line) in enumerate(iter_set.dataset):
            guid = "%s-%s" % (set_type, i)
            text_a = " ".join(line.text)
            text_b = ""
            label = line.label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            
        return examples


def get_test_data(processor, tokenizer, parameters, n_examples=50):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_mode = 'classification'
    task_name = parameters['TASK_NAME'].lower()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    eval_examples = processor.get_dev_examples()

    # subset to n_examples
    eval_examples = sample(eval_examples, n_examples)

    cached_eval_features_file = os.path.join(parameters['DATA_DIR'], 'dev_{0}_{1}_{2}'.format(
        list(filter(None, parameters['BERT_MODEL'].split('/'))).pop(),
                    str(parameters['MAX_SEQ_LENGTH']),
                    str(task_name)))
 
    eval_features = convert_examples_to_features(
            eval_examples, label_list, parameters['MAX_SEQ_LENGTH'], tokenizer, output_mode)

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
    
    #eval_sampler = SequentialSampler(eval_data)
    eval_sampler = RandomSampler(eval_data)  # Note that this sampler samples randomly
    
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=n_examples)

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    out_label_ids = None
  
    
    input_id_output = []
    input_mask_output = []
    segment_id_output = []
    label_id_output = []
    
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        return input_ids, input_mask, segment_ids, label_ids
    
    #return input_id_output,input_mask_output,segment_id_output,label_id_output
        
 
def get_onehot(T, size=30522):
    ohe = torch.zeros(T.shape[0], T.shape[1],size)
    
    for i, t in enumerate(T):
        for j, val in enumerate(t):
            ohe[i, j, val] = 1
    return ohe

def get_ids(batch):
    input_ids = batch[0]
    token_type_ids = batch[2]
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    return input_ids, position_ids, token_type_ids


def raw_batch_to_bert(batch):
    return batch[0], batch[2]


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode='classification'):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features