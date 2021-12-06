from torch import nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5LayerFF
import copy
import torch
def re_masking_to_max_size(concatenation_inputs, concatenation_masks, max_size=512):
    concatenation_masks = copy.deepcopy(concatenation_masks)
    N_CONCAT = len(concatenation_inputs)
    assert(N_CONCAT == len(concatenation_masks))

    S_BATCH = len(concatenation_inputs[0])
    assert(S_BATCH == sum([len(c) for c in concatenation_inputs])//N_CONCAT)

    # get the max_seq_len
    max_seq_len = 0
    for  b in range(S_BATCH):
        seq_len = 0
        for c in range(N_CONCAT):
            if seq_len > max_size:
                concatenation_masks[c][b][:] = 0
            elif seq_len + concatenation_masks[c][b].sum().item() > max_size:
                concatenation_masks[c][b][max_size - seq_len:] = 0
            seq_len += concatenation_masks[c][b].sum().item()
        
        if max_seq_len < seq_len:
            max_seq_len = seq_len
    
    re_masked_input =\
        concatenation_inputs[0].new(S_BATCH, max_seq_len, *concatenation_inputs[0][0][0].shape).zero_()
    re_masked_mask =\
        concatenation_masks[0].new(S_BATCH, max_seq_len).zero_()
    for  b in range(S_BATCH):
        seq_len = 0
        for c in range(N_CONCAT):
            current_seq_len = concatenation_masks[c][b].sum()
            if current_seq_len != 0:
                re_masked_input[b, seq_len:seq_len + current_seq_len] =\
                    concatenation_inputs[c][b, :current_seq_len]
                re_masked_mask[b, seq_len:seq_len + current_seq_len] = 1
                seq_len += current_seq_len

    return re_masked_input, re_masked_mask

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class T5PipelineModel(nn.Module):

    def __init__(self, model_name='t5-small', embedding_base=True):
        super().__init__()
        self.model_name = model_name
        self.outliner = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.feedforward = T5LayerFF(self.outliner.config)
        self.textualiser = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.embedding_base = True
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

    def add_tokens(self, tokens):
        for token in tokens:
            self.tokenizer.add_tokens(token)
        self.outliner.resize_token_embeddings(len(self.tokenizer))
        self.textualiser.resize_token_embeddings(len(self.tokenizer))

        
    def forward(self, input_ids=None, attention_mask=None
                , outline_labels=None, teacher_forcing_tokens_ids=None, teacher_forcing_attention_mask=None,
                labels=None,**kwargs):
        outline_output = None
        if teacher_forcing_tokens_ids is None:
            assert(outline_labels is not None)
            outline_output =\
                self.outliner(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=outline_labels,
                              output_hidden_states=True)
            textualiser_input_embeds_outlines = self.feedforward(outline_output['decoder_hidden_states'][-1])

            # To recompute according to attention_mask and outline_labels
        
        else: 
            textualiser_input_embeds_outlines =\
                self.textualiser.encoder.get_input_embeddings()(teacher_forcing_tokens_ids)
        textualiser_attention_mask_outlines =\
            attention_mask.new(outline_labels.shape[0:2])
        textualiser_attention_mask_outlines.zero_()

        textualiser_attention_mask_outlines[outline_labels!= 0] = 1
        textualiser_attention_mask_outlines = textualiser_attention_mask_outlines.long()

        textualiser_input_embeds_text =\
            self.textualiser.encoder.get_input_embeddings()(input_ids)   

        textualiser_input_embeds, textualiser_attention_mask =\
            re_masking_to_max_size((textualiser_input_embeds_outlines, textualiser_input_embeds_text),
                                    (textualiser_attention_mask_outlines,attention_mask)
                                    )

        pipeline_model_output = self.textualiser(inputs_embeds=textualiser_input_embeds, 
        
                                                 attention_mask=textualiser_attention_mask,
                                                 labels=labels)

        return outline_output, pipeline_model_output

    def generate(self, input_ids=None, attention_mask=None, max_length=512, length_penalty=1.0, 
                 num_beams=4, repetition_penalty=2.5, early_stopping=True, **kwargs):
        outlines = self.outliner.generate(input_ids=input_ids, attention_mask=attention_mask)[:, 1:].clone()
        textualiser_input_embeds_outlines =\
            self.outliner(input_ids=input_ids, attention_mask=attention_mask,
                          labels=outlines, output_hidden_states=True)['decoder_hidden_states'][-1]
        textualiser_input_embeds_outlines = self.feedforward(textualiser_input_embeds_outlines)
        textualiser_attention_mask_outlines =\
            attention_mask.new(outlines.shape[0:2])
        textualiser_attention_mask_outlines.zero_()
        textualiser_attention_mask_outlines[outlines != 0] = 1
        textualiser_attention_mask_outlines = textualiser_attention_mask_outlines.long()

        textualiser_input_embeds_text =\
            self.textualiser.encoder.get_input_embeddings()(input_ids)

        textualiser_input_embeds, textualiser_attention_mask =\
            re_masking_to_max_size((textualiser_input_embeds_outlines, textualiser_input_embeds_text),
                                    (textualiser_attention_mask_outlines, attention_mask)
                                    )
        decoder_input_ids =(
                    torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device) * self.textualiser.config.decoder_start_token_id
                )
        text = \
            self.textualiser.generate(inputs_embeds=textualiser_input_embeds,
                                      attention_mask=textualiser_attention_mask,
                                      decoder_input_ids=decoder_input_ids,
                                      max_length=max_length, length_penalty=length_penalty,
                                      num_beams=num_beams, repetition_penalty=repetition_penalty,
                                      early_stopping=early_stopping)

        return outlines, text



class T5RawModel(nn.Module):
    def __init__(self, model_name='t5-small', embedding_base=True):
        super().__init__()
        self.model_name = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

    def add_tokens(self, tokens):
        for token in tokens:
            self.tokenizer.add_tokens(token)
        self.model.resize_token_embeddings(len(self.tokenizer))

        
    def forward(self,**kwargs):
        return self.model.forward(**kwargs)

    def generate(self, input_ids=None, attention_mask=None, max_length=512, length_penalty=1.0, num_beams=4, repetition_penalty=2.5, early_stopping=True, **kwargs):
        return self.model.forward(input_ids=input_ids, attention_mask=attention_mask,
                                  max_length=max_length, length_penalty=length_penalty,
                                  num_beams=num_beams, repetition_penalty=repetition_penalty,
                                  early_stopping=True, **kwargs)

