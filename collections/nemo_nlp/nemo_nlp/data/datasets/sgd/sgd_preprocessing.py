import os

import nemo_nlp
import nemo_nlp.data.datasets.sgd.data_utils as data_utils


class SGDPreprocessor:
    """ 
    Convert the raw data to the standard format supported by
    StateTrackingSGDData.
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
                 bert_ckpt_dir):

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
            nf.logger.info("Start generating the dialogue examples.")
            data_utils._create_dialog_examples(processor,
                                               dial_file,
                                               dataset_split)
            nf.logger.info("Finish generating the dialogue examples.")


        schema_embedding_file = os.path.join(schema_embedding_dir,
        f"{dataset_split}_pretrained_schema_embedding.npy")

        # Generate the schema embeddings if needed or specified.
        if not os.path.exists(schema_embedding_file) or overwrite_schema_emb_file:
            nf.logger.info("Start generating the schema embeddings.")
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

            def concatenate(lists):
                return np.concatenate([t.cpu() for t in lists])


            def get_preds(logits):
                return np.argmax(logits, 1)

            hidden_states = [concatenate(tensors) for tensors in evaluated_tensors]
            emb_datalayer.dataset.save_embeddings(hidden_states,
                                                  schema_embedding_file)
            nf.logger.info(f"The schema embeddings saved at {schema_embedding_file}")
            nf.logger.info("Finish generating the schema embeddings.")