"""
This code were adapted from 
https://github.com/google-research/google-research/tree/master/schema_guided_dst
"""

def if_exist(outfold, files):
	if not os.path.exists(outfold):
		return False
	for file in files:
		if not os.path.exists(f'{outfold}/{file}'):
			return False
	return True


class SGDDataset:
	""" Convert the raw data to the standard format supported by
	StateTrackingSGDData.
	TODO: Update here

	"""

	def __init__(self,
				 data_dir,
				 tokenizer,
				 task_name,
				 dataset_split,
				 do_lower_case=False,
				 dataset_name='default',
				 none_slot_label='O',
				 pad_label=-1,
				 max_seq_length=50,
				 modes=['train', 'eval'],
				 log_data_warnings=False):
		if dataset_name == 'sgd':
			self.data_dir = process_sgd(data_dir,
										do_lower_case,
										dataset_name=dataset_name,
										max_seq_length=max_seq_length,
										task_name=task_name,
										tokenizer=tokenizer,
										dataset_split=dataset_split,
										modes=modes,
										log_data_warnings=log_data_warnings)
		else:
			if not if_exist(data_dir, ['dialogues.tsv']):
				raise FileNotFoundError(
					"Make sure that your data follows the standard format "
					"supported by StateTrackerDataset. Your data must "
					"contain dialogues.tsv.")
			self.data_dir = data_dir

	# Changed here
	# self.intent_dict_file = self.data_dir + '/dict.intents.csv'
	# self.slot_dict_file = self.data_dir + '/dict.slots.csv'
	# self.num_intents = len(get_vocab(self.intent_dict_file))
	# slots = label2idx(self.slot_dict_file)
	# self.num_slots = len(slots)
	#
	# for mode in ['train', 'test', 'eval']:
	#
	# 	if not if_exist(self.data_dir, [f'{mode}.tsv']):
	# 		logger.info(f' Stats calculation for {mode} mode'
	# 					f' is skipped as {mode}.tsv was not found.')
	# 		continue
	#
	# 	slot_file = f'{self.data_dir}/{mode}_slots.tsv'
	# 	with open(slot_file, 'r') as f:
	# 		slot_lines = f.readlines()
	#
	# 	input_file = f'{self.data_dir}/{mode}.tsv'
	# 	with open(input_file, 'r') as f:
	# 		input_lines = f.readlines()[1:]  # Skipping headers at index 0
	#
	# 	if len(slot_lines) != len(input_lines):
	# 		raise ValueError(
	# 			"Make sure that the number of slot lines match the "
	# 			"number of intent lines. There should be a 1-1 "
	# 			"correspondence between every slot and intent lines.")
	#
	# 	dataset = list(zip(slot_lines, input_lines))
	#
	# 	raw_slots, queries, raw_intents = [], [], []
	# 	for slot_line, input_line in dataset:
	# 		slot_list = [int(slot) for slot in slot_line.strip().split()]
	# 		raw_slots.append(slot_list)
	# 		parts = input_line.strip().split()
	# 		raw_intents.append(int(parts[-1]))
	# 		queries.append(' '.join(parts[:-1]))
	#
	# 	infold = input_file[:input_file.rfind('/')]
	#
	# 	logger.info(f'Three most popular intents during {mode}ing')
	# 	total_intents, intent_label_freq = get_label_stats(
	# 		raw_intents, infold + f'/{mode}_intent_stats.tsv')
	# 	merged_slots = itertools.chain.from_iterable(raw_slots)
	#
	# 	logger.info(f'Three most popular slots during {mode}ing')
	# 	slots_total, slots_label_freq = get_label_stats(
	# 		merged_slots, infold + f'/{mode}_slot_stats.tsv')
	#
	# 	if mode == 'train':
	# 		self.slot_weights = calc_class_weights(slots_label_freq)
	# 		logger.info(f'Slot weights are - {self.slot_weights}')
	#
	# 		self.intent_weights = calc_class_weights(intent_label_freq)
	# 		logger.info(f'Intent weights are - {self.intent_weights}')
	#
	# 	logger.info(f'Total intents - {total_intents}')
	# 	logger.info(f'Intent label frequency - {intent_label_freq}')
	# 	logger.info(f'Total Slots - {slots_total}')
	# 	logger.info(f'Slots label frequency - {slots_label_freq}')
	#
	# if pad_label != -1:
	# 	self.pad_label = pad_label
	# else:
	# 	if none_slot_label not in slots:
	# 		raise ValueError(f'none_slot_label {none_slot_label} not '
	# 						 f'found in {self.slot_dict_file}.')
	# 	self.pad_label = slots[none_slot_label]