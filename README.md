# Enzymatic Transformer


This repo complements the "[Predicting Enzymatic Reactions with a Molecular Transformer](https://chemrxiv.org/articles/preprint/Predicting_Enzymatic_Reactions_with_a_Molecular_Transformer/13161359/1)" publication

## Requirements

### Specific versions used:
- Python: 3.6.10
- Torch: 1.5.1
- TorchText: 0.6.1
- OpenNMT: 1.1.1
- RDKit: 2017.09.1

### Conda Environment Setup

```bash
conda create -n enztrans_test python=3.6
conda activate enztrans_test
conda install -c rdkit rdkit=2017.09.1 -y
conda install -c pytorch pytorch=1.5.1 -y
git clone https://github.com/reymond-group/OpenNMT-py.git
cd OpenNMT-py
git checkout Enzymatic_Transformer
pip install -e .
```

## Quickstart

The training and evaluation was performed using [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). The full documentation of the OpenNMT can be found [here](https://opennmt.net/OpenNMT-py/).

### Step 1: Tokenization 

The reaction SMILES are tokenized using the tokenization function available from the Molecular Transformer [here](https://github.com/pschwllr/MolecularTransformer)

Enzyme sentences are tokenized using the Hugging Face tokenizers available [here](https://github.com/huggingface/tokenizers/tree/master/bindings/python#build-your-own). The custom tokenizer can be build from a file containing the list of sentences using the following commands:

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Initialize a tokenizer
tokenizer2 = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer2.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer2.decoder = decoders.ByteLevel()
tokenizer2.post_processor = processors.ByteLevel(trim_offsets=True)

# And then train
trainer = trainers.BpeTrainer(vocab_size=9000, min_frequency=2, limit_alphabet=55, special_tokens=['ase', 'hydro', 'mono', 'cyclo', 'thermo', 'im'])
tokenizer2.train(trainer, ["list_of_sentences.txt"])
```

Then, sentences of the dataset are tokenized using the following function:

```python
def enzyme_sentence_tokenizer(sentence):
    '''
    Tokenize a sentenze, optimized for enzyme-like descriptions & names
    '''
    encoded = tokenizer2.encode(sentence)
    my_list = [item for item in encoded.tokens if 'Ġ' != item]
    my_list = [item.replace('Ġ', '_') for item in my_list]
    my_list = ' '.join(my_list)
    return my_list
```

### Step 2: Preprocess the data

```bash
DATASET=data/uspto_dataset
DATASET_TRANSFER=data/transfer_dataset

preprocess.py -train_ids ENZR ST_sep_aug \
	-train_src DATADIR/src_train.txt $DATASET_TRANSFER/src-train.txt \
	-train_tgt DATADIR/tgt_train.txt $DATASET_TRANSFER/tgt-train.txt \
	-valid_src DATADIR/src_val.txt -valid_tgt $DATASET_TRANSFER/multi_task /tgt_val.txt \
	-save_data DATADIR/Preprocessed \-src_seq_length 3000 -tgt_seq_length 3000 \
	-src_vocab_size 3000 -tgt_vocab_size 3000 \-share_vocab -lower
```

### Step 3: Training of the model

The Enzymatic Transformer was trained using the following parameters:

Multi-task transfer learning:

```bash
WEIGHT1=1
WEIGHT2=9

train.py -data DATADIR/Preprocessed \
	-save_model ENZR_MTL -seed 42 -train_steps 200000 -param_init 0 \
	-param_init_glorot  -max_generator_batches 32 -batch_size 6144 \
	-batch_type tokens -normalization tokens -max_grad_norm 0 -accum_count 4 \
	-optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam \
	-warmup_steps 8000 -learning_rate 4 -label_smoothing 0.0 -layers 4 \
	-rnn_size 384 -word_vec_size 384 \
	-encoder_type transformer -decoder_type transformer \
	-dropout 0.1 -position_encoding -global_attention general \
	-global_attention_function softmax -self_attn_type scaled-dot \
	-heads 8 -transformer_ff 2048 \
	-data_ids ENZR ST_sep_aug -data_weights WEIGHT1 WEIGHT2 \
	-valid_steps 5000 -valid_batch_size 4 -early_stopping_criteria accuracy \
```





### Step 4: Model prediction


A reaction can be predicted after tokenization using the following command:

```bash
translate.py -model model_uspto_ENZR_multitask.pt \
	-src DATASET/src_test.txt \
	-output predictions.txt \
	-batch_size 64 -replace_unk -max_length 1000 \
	-log_probs -beam_size 5 -n_best 5 \

```

## Citation

### Enzymatic Transformer:

```bash
@article{kreutter_predicting_2020,
	title = {Predicting {Enzymatic} {Reactions} with a {Molecular} {Transformer}},
	author = {Kreutter, David and Schwaller, Philippe and Reymond, Jean-Louis},
	url = {/articles/preprint/Predicting_Enzymatic_Reactions_with_a_Molecular_Transformer/13161359/1},
	doi = {10.26434/chemrxiv.13161359.v1},
	urldate = {2020-10-30},
	month = oct,
	year = {2020},
	note = {Publisher: ChemRxiv}
}
```

### Original OpenNMT-py:

If you reuse this code please also cite the underlying code framework:

[OpenNMT: Neural Machine Translation Toolkit](https://arxiv.org/pdf/1805.11462.pdf)

[OpenNMT technical report](https://www.aclweb.org/anthology/P17-4012/)

```bash
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```








