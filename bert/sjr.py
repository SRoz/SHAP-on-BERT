import pytorch_pretrained_bert
import torch
from torch import nn

class HalfEmbeddings(pytorch_pretrained_bert.modeling.BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, parent):
        super(pytorch_pretrained_bert.modeling.BertEmbeddings, self).__init__()
        self.word_embeddings = parent.word_embeddings 
        self.position_embeddings = parent.position_embeddings 
        self.token_type_embeddings = parent.token_type_embeddings


    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        return embeddings, words_embeddings, position_embeddings, token_type_embeddings

class SecondHalfEmbedding(pytorch_pretrained_bert.modeling.BertEmbeddings):
    """Construct the embeddings from the OHE matricies.
    """
    def __init__(self, parent, config=None):
        super(pytorch_pretrained_bert.modeling.BertEmbeddings, self).__init__()
        self.word_embedding_matrix = parent.word_embeddings._parameters['weight']
        self.position_embedding_matrix = parent.position_embeddings._parameters['weight']
        self.token_type_embedding_matrix = parent.token_type_embeddings._parameters['weight']

        self.LayerNorm = parent.LayerNorm 
        self.dropout = parent.dropout

    def forward(self, word_ohe, position_ohe, token_ohe): 
        words_embeddings = torch.matmul(word_ohe, self.word_embedding_matrix)
        position_embeddings = torch.matmul(position_ohe, self.position_embedding_matrix)
        token_type_embeddings = torch.matmul(token_ohe, self.token_type_embedding_matrix)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSJRModel(nn.Module):#BertPreTrainedModel):
    def __init__(self, parent, INPUT_ID_EX, config=None, output_attentions=False, keep_multihead_output=False):
        super(BertSJRModel, self).__init__()
        
        self.num_labels = parent.num_labels
        self.output_attentions = output_attentions
        self.parameters = parent.parameters
        self.config = parent.config
        
        self.INPUT_ID_EX = INPUT_ID_EX
        #self.eval = parent.eval
        
        self.embeddings = SecondHalfEmbedding(parent.bert.embeddings)
        
        self.encoder = parent.bert.encoder
        self.pooler = parent.bert.pooler
        
        self.dropout = parent.dropout
        self.classifier = parent.classifier
        #self.apply(self.init_bert_weights)
        
    
    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_multihead_outputs(self):
        """ Gather all multi-head outputs.
            Return: list (layers) of multihead module outputs with gradients
        """
        return [layer.attention.self.multihead_output for layer in self.encoder.layer]

    def forward(self, word_ohe, position_ohe, token_ohe, labels=None, attention_mask=None, 
                output_all_encoded_layers=True, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(self.INPUT_ID_EX)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length
        
        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(word_ohe, position_ohe, token_ohe)
        
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)#,
                                      #head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if self.output_attentions:
            outputs = all_attentions, encoded_layers, pooled_output
        outputs = encoded_layers, pooled_output
    
        # FROM BELOW
        if self.output_attentions:
            all_attentions, _, pooled_output = outputs
        else:
            _, pooled_output = outputs
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        elif self.output_attentions:
            return all_attentions , logits
        return logits