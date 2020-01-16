"""
This code were adapted from 
https://github.com/google-research/google-research/tree/master/schema_guided_dst
"""

import argparse
import math
import os

import numpy as np
from pytorch_transformers import BertTokenizer

import nemo

import nemo_nlp
import nemo_nlp.data.datasets.sgd.data_utils as data_utils
from nemo_nlp.data.datasets.sgd import tokenization
from nemo_nlp.utils.callbacks.joint_intent_slot import \
	eval_iter_callback, eval_epochs_done_callback

# Parsing arguments
parser = argparse.ArgumentParser(description='Schema_guided_dst')

# BERT based utterance encoder related arguments
parser.add_argument("--bert_ckpt_dir", default=None, type=str,
					required=True,
					help="Directory containing pre-trained BERT checkpoint.")
parser.add_argument("--do_lower_case", default=False, type=bool,
				    help="Whether to lower case the input text. Should be True for uncased "
				    "models and False for cased models.")
parser.add_argument("--preserve_unused_tokens", default=False, type=bool,
				    help="If preserve_unused_tokens is True, Wordpiece tokenization will not "
				    "be applied to words in the vocab.")
parser.add_argument("--max_seq_length", default=80, type=int,
				    help="The maximum total input sequence length after WordPiece tokenization. "
				    "Sequences longer than this will be truncated, and sequences shorter "
				    "than this will be padded.")
parser.add_argument("--dropout_rate", default=0.1, type=float,
                   help="Dropout rate for BERT representations.")

# Hyperparameters and optimization related flags.
parser.add_argument("--train_batch_size", default=32, type=int,
					 help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int,
					 help="Total batch size for eval.")
parser.add_argument("--predict_batch_size", default=8, type=int,
					 help="Total batch size for predict.")
parser.add_argument("--learning_rate", default=1e-4, type=float, 
					 help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=80.0, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
	                help="Proportion of training to perform linear learning rate warmup for. "
                    "E.g., 0.1 = 10% of training.")
parser.add_argument("--save_checkpoints_steps", default=1000, type=int,
                    help="How often to save the model checkpoint.")

# Input and output paths and other flags.
parser.add_argument("--task_name", default="dstc8_single_domain", type=str,
				    choices=data_utils.FILE_RANGES.keys(),
                    help="The name of the task to train.")

parser.add_argument("--data_dir", type=str, required=True,
				    help="Directory for the downloaded DSTC8 data, which contains the dialogue files"
				    " and schema files of all datasets (eg train, dev)")

parser.add_argument("--run_mode", default="train", type=str,
					choices=["train", "predict"],
                    help="The mode to run the script in.")

parser.add_argument("--work_dir", type=str, required=True,
                    help="The output directory where the model checkpoints will be written.")

parser.add_argument("--schema_embedding_dir", type=str, required=True,
    				help="Directory where .npy file for embedding of entities (slots, values,"
       				" intents) in the dataset_split's schema are stored.")

parser.add_argument("--dialogues_example_dir", type=str, required=True,
    				help="Directory where tf.record of DSTC8 dialogues data are stored.")

parser.add_argument("--dataset_split", type=str, required=True,
					choices=["train", "dev", "test"],
                    help="Dataset split for training / prediction.")

parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--amp_opt_level", default="O0",
                    type=str, choices=["O0", "O1", "O2"])

# flags.DEFINE_string(
#     "eval_ckpt", "",
#     "Comma separated numbers, each being a step number of model checkpoint"
#     " which makes predictions.")

# flags.DEFINE_bool(
#     "overwrite_dial_file", False,
#     "Whether to generate a new Tf.record file saving the dialogue examples.")

# flags.DEFINE_bool(
#     "overwrite_schema_emb_file", False,
#     "Whether to generate a new schema_emb file saving the schemas' embeddings.")

# flags.DEFINE_bool(
#     "log_data_warnings", False,
#     "If True, warnings created using data processing are logged.")

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

task_name = args.task_name
vocab_file = os.path.join(args.bert_ckpt_dir, "vocab.txt")

if not os.path.exists(vocab_file):
    raise ValueError(f'vocab_file.txt not found at {vocab_file}')

work_dir = f'{args.work_dir}/{args.task_name.upper()}'
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
								   local_rank=args.local_rank,
								   optimization_level=args.amp_opt_level,
								   log_dir=work_dir,
								   create_tb_writer=True,
								   files_to_copy=[__file__],
								   add_time_to_log_dir=True)

processor = data_utils.Dstc8DataProcessor(
      args.data_dir,
      train_file_range=data_utils.FILE_RANGES[task_name]["train"],
      dev_file_range=data_utils.FILE_RANGES[task_name]["dev"],
      test_file_range=data_utils.FILE_RANGES[task_name]["test"],
      vocab_file=vocab_file,
      do_lower_case=args.do_lower_case,
      preserve_unused_tokens=args.preserve_unused_tokens,
      max_seq_length=args.max_seq_length)

# Generate the dialogue examples if needed or specified.
dial_file_name = "{}_{}_examples.processed".format(task_name,
                                                   args.dataset_split)
dial_file = os.path.join(args.dialogues_example_dir, dial_file_name)

if not os.path.exists(args.dialogues_example_dir):
    os.makedirs(args.dialogues_example_dir)
if not os.path.exists(dial_file):
	nf.logger.info("Start generating the dialogue examples.")

data_utils._create_dialog_examples(processor, dial_file, args.dataset_split)
nf.logger.info("Finish generating the dialogue examples.")

# # Generate the schema embeddings if needed or specified.
# bert_init_ckpt = os.path.join(args.bert_ckpt_dir, "bert_model.ckpt")
# tokenization.validate_case_matches_checkpoint(
#   do_lower_case=args.do_lower_case, init_checkpoint=bert_init_ckpt)

# bert_config = modeling.BertConfig.from_json_file(
#   os.path.join(FLAGS.bert_ckpt_dir, "bert_config.json"))
# if FLAGS.max_seq_length > bert_config.max_position_embeddings:
# raise ValueError(
#     "Cannot use sequence length %d because the BERT model "
#     "was only trained up to sequence length %d" %
#     (FLAGS.max_seq_length, bert_config.max_position_embeddings))



# """ Load the pretrained BERT parameters
# See the list of pretrained models, call:
# nemo_nlp.huggingface.BERT.list_pretrained_models()
# """
# pretrained_bert_model = nemo_nlp.huggingface.BERT(
# 	pretrained_model_name=args.pretrained_bert_model, factory=nf)
# hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
# tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

# task_name = args.task_name.lower()
# if task_name not in sgd_utils.FILE_RANGES:
# 	raise ValueError(f'Task not found: {task_name}')

# data_desc = SGDDataset(args.data_dir,
# 									 tokenizer,
# 									 task_name=task_name,
# 									 dataset_split=args.dataset_split,
# 									 do_lower_case=args.do_lower_case,
# 									 dataset_name=args.dataset_name,
# 									 none_slot_label=args.none_slot_label,
# 									 pad_label=args.pad_label,
# 									 max_seq_length=args.max_seq_length)

# # Create sentence classification loss on top
# classifier = nemo_nlp.JointIntentSlotClassifier(
# 	hidden_size=hidden_size,
# 	num_intents=data_desc.num_intents,
# 	num_slots=data_desc.num_slots,
# 	dropout=args.fc_dropout)

# loss_fn = nemo_nlp.JointIntentSlotLoss(num_slots=data_desc.num_slots)


# def create_pipeline(num_samples=-1,
# 					batch_size=32,
# 					num_gpus=1,
# 					local_rank=0,
# 					mode='train'):
# 	nf.logger.info(f"Loading {mode} data...")
# 	data_file = f'{data_desc.data_dir}/{mode}.tsv'
# 	slot_file = f'{data_desc.data_dir}/{mode}_slots.tsv'
# 	shuffle = args.shuffle_data if mode == 'train' else False

# 	data_layer = nemo_nlp.BertJointIntentSlotDataLayer(
# 		input_file=data_file,
# 		slot_file=slot_file,
# 		pad_label=data_desc.pad_label,
# 		tokenizer=tokenizer,
# 		max_seq_length=args.max_seq_length,
# 		num_samples=num_samples,
# 		shuffle=shuffle,
# 		batch_size=batch_size,
# 		num_workers=0,
# 		local_rank=local_rank,
# 		ignore_extra_tokens=args.ignore_extra_tokens,
# 		ignore_start_end=args.ignore_start_end
# 	)

# 	ids, type_ids, input_mask, loss_mask, \
# 	subtokens_mask, intents, slots = data_layer()
# 	data_size = len(data_layer)

# 	if data_size < batch_size:
# 		nf.logger.warning("Batch_size is larger than the dataset size")
# 		nf.logger.warning("Reducing batch_size to dataset size")
# 		batch_size = data_size

# 	steps_per_epoch = math.ceil(data_size / (batch_size * num_gpus))
# 	nf.logger.info(f"Steps_per_epoch = {steps_per_epoch}")

# 	hidden_states = pretrained_bert_model(input_ids=ids,
# 										  token_type_ids=type_ids,
# 										  attention_mask=input_mask)

# 	intent_logits, slot_logits = classifier(hidden_states=hidden_states)

# 	loss = loss_fn(intent_logits=intent_logits,
# 				   slot_logits=slot_logits,
# 				   loss_mask=loss_mask,
# 				   intents=intents,
# 				   slots=slots)

# 	if mode == 'train':
# 		tensors_to_evaluate = [loss, intent_logits, slot_logits]
# 	else:
# 		tensors_to_evaluate = [intent_logits, slot_logits, intents,
# 							   slots, subtokens_mask]

# 	return tensors_to_evaluate, loss, steps_per_epoch, data_layer


# train_tensors, train_loss, steps_per_epoch, _ = create_pipeline(
# 	args.num_train_samples,
# 	batch_size=args.batch_size,
# 	num_gpus=args.num_gpus,
# 	local_rank=args.local_rank,
# 	mode=args.train_file_prefix)
# eval_tensors, _, _, data_layer = create_pipeline(
# 	args.num_eval_samples,
# 	batch_size=args.batch_size,
# 	num_gpus=args.num_gpus,
# 	local_rank=args.local_rank,
# 	mode=args.eval_file_prefix)

# # Create callbacks for train and eval modes
# train_callback = nemo.core.SimpleLossLoggerCallback(
# 	tensors=train_tensors,
# 	print_func=lambda x: str(np.round(x[0].item(), 3)),
# 	tb_writer=nf.tb_writer,
# 	get_tb_values=lambda x: [["loss", x[0]]],
# 	step_freq=steps_per_epoch)

# eval_callback = nemo.core.EvaluatorCallback(
# 	eval_tensors=eval_tensors,
# 	user_iter_callback=lambda x, y: eval_iter_callback(
# 		x, y, data_layer),
# 	user_epochs_done_callback=lambda x: eval_epochs_done_callback(
# 		x, f'{nf.work_dir}/graphs'),
# 	tb_writer=nf.tb_writer,
# 	eval_step=steps_per_epoch)

# # Create callback to save checkpoints
# ckpt_callback = nemo.core.CheckpointCallback(
# 	folder=nf.checkpoint_dir,
# 	epoch_freq=args.save_epoch_freq,
# 	step_freq=args.save_step_freq)

# lr_policy_fn = get_lr_policy(args.lr_policy,
# 							 total_steps=args.num_epochs * steps_per_epoch,
# 							 warmup_ratio=args.lr_warmup_proportion)

# nf.train(tensors_to_optimize=[train_loss],
# 		 callbacks=[train_callback, eval_callback, ckpt_callback],
# 		 lr_policy=lr_policy_fn,
# 		 optimizer=args.optimizer_kind,
# 		 optimization_params={"num_epochs": args.num_epochs,
# 							  "lr": args.lr,
# 							  "weight_decay": args.weight_decay})
