{
    
  local num_filters = 150,
  local dropout = 0.1,
  local lr = 0.0001,
  local embedding_dim = 1024,
  local ngram_filter_sizes = [2, 3, 4, 5],
  local classifier_feedforward_input_dim  = 4*150*2,

  local data_dir = "./",
  local robert_archive = "robert_model_path",
  "dataset_reader": {
    "type": "matching_reader_ernie",
    "max_seq_len":256,
    "pretrained_model": robert_archive+"/vocab.txt",
    "token_indexers": {
      "tokens": {
         "type": "bert-pretrained",
          "pretrained_model": robert_archive+"/vocab.txt",
          "do_lowercase": false,
          "use_starting_offsets": true
      }
    }
  },

  "train_data_path": data_dir+ "el_train.jsonl",
  "validation_data_path": data_dir+ "el_dev.jsonl",
  "model": {
    "type": "bimpm_matching_fea_task",
    "dropout": dropout,
    "gamma": 0.5,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
          "tokens": ["tokens", "tokens-offsets"]
        },
        "token_embedders": {
            "tokens": {
                "type": "bert-pretrained",
                "pretrained_model": robert_archive,
            }
        }
    },
    "pos_tag_encoder": {
        "embedding": {
              "embedding_dim": 40,
              "vocab_namespace": "pos_tags"
        },
        "pos_vocab_namespace": "pos_tags"
    },
    "matcher_word": {
        "hidden_dim": embedding_dim,
        "with_full_match": true,
        "is_forward":true
    },
    "matcher_forward1": {
       "hidden_dim": 200,
       "with_full_match":true,
       "is_forward":true
    },
    "matcher_backward1": {
      "hidden_dim": 200,
      "with_full_match":true,
      "is_forward":false
    },
    "matcher_forward2": {
       "hidden_dim": 200,
       "with_full_match":true,
       "is_forward":true
    },
    "matcher_backward2": {
      "hidden_dim": 200,
      "with_full_match":true,
      "is_forward":false
    },
    "encoder1": {
        "type": "lstm",
        "input_size": embedding_dim,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": dropout,
        "bidirectional": true
    },
    "encoder2": {
        "type": "lstm",
        "input_size": 400,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": dropout,
        "bidirectional": true
    },
    "aggregator": {
          "type": "cnn",
          "num_filters":num_filters,
          "ngram_filter_sizes": ngram_filter_sizes,
          "embedding_dim": 525
    },
    "classifier_feedforward": {
        "input_dim": classifier_feedforward_input_dim,
        "num_layers": 1,
        "hidden_dims": 1,
        "activations": "linear"
    },
    "fea_classifier_feedforward": {
        "input_dim": 3,
        "num_layers": 1,
        "hidden_dims": 2,
        "activations": "linear"
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 8
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": lr
    },
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 10,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": [0]
  }
}