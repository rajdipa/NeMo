import os

import nemo
import nemo_nlp
from  nemo_nlp.utils import nlp_utils
from nemo_nlp.data.datasets.sgd import data_utils


class SGDPreprocessor:
    """ 
    Convert the raw data to the standard format supported by
    StateTrackingSGDData.
    
    Args:
        data_dir (str) - Directory for the downloaded DSTC8 data, which contains
            the dialogue files and schema files of all datasets (eg train, dev)
        dialogues_example_dir (str) - Directory where preprocessed DSTC8 dialogues are stored
        schema_embedding_dir (str) - Directory where .npy file for embedding of
            entities (slots, values, intents) in the dataset_split's
            schema are stored.
        task_name (str) - The name of the task to train
        vocab_file (str) - The path to BERT vocab file
        do_lower_case - (bool) - Whether to lower case the input text.
            Should be True for uncased models and False for cased models.
        max_seq_length (int) - The maximum total input sequence length after
            WordPiece tokenization. Sequences longer than this will be
            truncated, and sequences shorter than this will be padded."
        tokenizer - tokenizer
        bert_model - pretrained BERT model
        dataset_split (str) - Dataset split for training / prediction (train/dev/test)
        overwrite_dial_file (bool) - Whether to generate a new file saving
            the dialogue examples overwrite_schema_emb_file,
        bert_ckpt_dir (str) - Directory containing pre-trained BERT checkpoint
        nf - NeuralModuleFactory
    """
    def __init__(self,
                 data_dir,
                 dialogues_example_dir,
                 schema_embedding_dir,
                 task_name,
                 vocab_file,
                 do_lower_case,
                 max_seq_length,
                 tokenizer,
                 bert_model,
                 dataset_split,
                 overwrite_dial_file,
                 overwrite_schema_emb_file,
                 bert_ckpt_dir,
                 nf):

        processor = data_utils.Dstc8DataProcessor(
              data_dir,
              train_file_range=data_utils.FILE_RANGES[task_name]["train"],
              dev_file_range=data_utils.FILE_RANGES[task_name]["dev"],
              test_file_range=data_utils.FILE_RANGES[task_name]["test"],
              vocab_file=vocab_file,
              do_lower_case=do_lower_case,
              tokenizer=tokenizer,
              max_seq_length=max_seq_length)

        # Generate the dialogue examples if needed or specified.
        dial_file_name = f"{task_name}_{dataset_split}_examples.processed"
        dial_file = os.path.join(dialogues_example_dir,
                                 dial_file_name)

        if not os.path.exists(dialogues_example_dir):
            os.makedirs(dialogues_example_dir)
        if not os.path.exists(dial_file) or overwrite_dial_file:
            nemo.logging.info("Start generating the dialogue examples.")
            data_utils._create_dialog_examples(processor,
                                               dial_file,
                                               dataset_split)
            nemo.logging.info("Finish generating the dialogue examples.")


        schema_embedding_file = os.path.join(schema_embedding_dir,
        f"{dataset_split}_pretrained_schema_embedding.npy")

        # Generate the schema embeddings if needed or specified.
        if not os.path.exists(schema_embedding_file) or overwrite_schema_emb_file:
            nemo.logging.info("Start generating the schema embeddings.")
            # create schema embedding if no file exists
            schema_json_path = os.path.join(data_dir, 
                                            dataset_split,
                                            "schema.json")

            emb_datalayer = nemo_nlp.BertInferDataLayer(dataset_type='SchemaEmbeddingDataset',
                                                        tokenizer=tokenizer,
                                                        max_seq_length=max_seq_length,
                                                        input_file=schema_json_path)
            
            input_ids, input_mask, input_type_ids = emb_datalayer()
            hidden_states = bert_model(input_ids=input_ids,
                                       token_type_ids=input_type_ids,
                                       attention_mask=input_mask)

            evaluated_tensors = nf.infer(tensors=[hidden_states],
                                         checkpoint_dir=bert_ckpt_dir)



            hidden_states = [nlp_utils.concatenate(tensors) for tensors in evaluated_tensors]
            emb_datalayer.dataset.save_embeddings(hidden_states,
                                                  schema_embedding_file)
            nemo.logging.info(f"The schema embeddings saved at {schema_embedding_file}")
            nemo.logging.info("Finish generating the schema embeddings.")