import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import bert_utils
import modeling 
import tokenization
import json
import tqdm
import jsonlines
import importlib
import os
importlib.reload(bert_utils)
import argparse
import webbrowser


parser = argparse.ArgumentParser()
parser.add_argument("temp")
args = parser.parse_args()
nq_test_file = args.temp
# print(nq_test_file)

with open('bert_config.json','r') as f:
    config = json.load(f)

    
if nq_test_file == 'sample_best1.jsonl':
    webbrowser.open('https://en.wikipedia.org/wiki/Karnataka_Police')
    
if nq_test_file == 'sample_best2.jsonl':
    webbrowser.open('https://en.wikipedia.org/wiki/Vegas_Golden_Knights')
    
if nq_test_file == 'sample_bad1.jsonl':
    webbrowser.open('https://en.wikipedia.org/wiki/The_Block_(season_8)')
    
if nq_test_file == 'sample_bad2.jsonl':
    webbrowser.open('https://en.wikipedia.org/wiki/Interstate_485')
    
    
    
class TDense(tf.keras.layers.Layer):
    def __init__(self,
                 output_size,
                 kernel_initializer=None,
                 bias_initializer="zeros",
                **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
    def build(self,input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `TDense` layer with "
                          "non-floating point (and non-complex) "
                          "dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError("The last dimension of the inputs to "
                           "`TDense` should be defined. "
                           "Found `None`.")
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.output_size,last_dim],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            "bias",
            shape=[self.output_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
            trainable=True)
        super(TDense, self).build(input_shape)
    def call(self,x):
        return tf.matmul(x,self.kernel,transpose_b=True)+self.bias
    
def mk_model(config):
    seq_len = config['max_position_embeddings']
    unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_id')
    input_ids   = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_ids')
    input_mask  = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_mask')
    segment_ids = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='segment_ids')
    BERT = modeling.BertModel(config=config,name='bert')
    pooled_output, sequence_output = BERT(input_word_ids=input_ids,
                                          input_mask=input_mask,
                                          input_type_ids=segment_ids)
    
    logits = TDense(2,name='logits')(sequence_output)
    start_logits,end_logits = tf.split(logits,axis=-1,num_or_size_splits= 2,name='split')
    start_logits = tf.squeeze(start_logits,axis=-1,name='start_squeeze')
    end_logits   = tf.squeeze(end_logits,  axis=-1,name='end_squeeze')
    
    ans_type      = TDense(5,name='ans_type')(pooled_output)
    return tf.keras.Model([input_ for input_ in [unique_id,input_ids,input_mask,segment_ids] 
                           if input_ is not None],
                          [unique_id,start_logits,end_logits,ans_type],
                          name='bert-baseline')    

small_config = config.copy()
small_config['vocab_size']=16
small_config['hidden_size']=64
small_config['max_position_embeddings'] = 32
small_config['num_hidden_layers'] = 4
small_config['num_attention_heads'] = 4
small_config['intermediate_size'] = 256

model= mk_model(config)

cpkt = tf.train.Checkpoint(model=model)
cpkt.restore('model_cpkt-1').assert_consumed()

eval_records = "nq-test-new.tfrecords"
# nq_test_file = 'sample_best1.jsonl'

if os.path.exists(eval_records):
    eval_writer = bert_utils.FeatureWriter(
        filename=os.path.join(eval_records),
        is_training=False)

    tokenizer = tokenization.FullTokenizer(vocab_file='vocab-nq.txt', 
                                           do_lower_case=True)

    features = []
    convert = bert_utils.ConvertExamples2Features(tokenizer=tokenizer,
                                                   is_training=False,
                                                   output_fn=eval_writer.process_feature,
                                                   collect_stat=False)

    n_examples = 0
    for examples in bert_utils.nq_examples_iter(input_file=nq_test_file, 
                                           is_training=False,
                                           tqdm=None):
        for example in examples:
            n_examples += convert(example)

    eval_writer.close()
#     print('number of test examples: %d, written to file: %d' % (n_examples,eval_writer.num_features))


seq_length = bert_utils.FLAGS.max_seq_length #config['max_position_embeddings']
name_to_features = {
      "unique_id": tf.io.FixedLenFeature([], tf.int64),
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
  }


def _decode_record(record, name_to_features=name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if name != 'unique_id': #t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = t

    return example

def _decode_tokens(record):
    return tf.io.parse_single_example(serialized=record, 
                                      features={
                                          "unique_id": tf.io.FixedLenFeature([], tf.int64),
                                          "token_map" :  tf.io.FixedLenFeature([seq_length], tf.int64)
                                      })
      

raw_ds = tf.data.TFRecordDataset(eval_records)
token_map_ds = raw_ds.map(_decode_tokens)
decoded_ds = raw_ds.map(_decode_record)
ds = decoded_ds.batch(batch_size=16,drop_remainder=False)

result=model.predict_generator(ds,verbose=0)

np.savez_compressed('bert-joint-baseline-output.npz',
                    **dict(zip(['uniqe_id','start_logits','end_logits','answer_type_logits'],
                               result)))

all_results = [bert_utils.RawResult(*x) for x in zip(*result)]

# print ("Going to candidates file")
candidates_dict = bert_utils.read_candidates(nq_test_file)

# print ("setting up eval features")
eval_features = list(token_map_ds)

# print ("compute_pred_dict")
nq_pred_dict = bert_utils.compute_pred_dict(candidates_dict, 
                                       eval_features,
                                       all_results,
                                      tqdm=None)

predictions_json = {"predictions": list(nq_pred_dict.values())}
# print ("writing json")
with tf.io.gfile.GFile('predictions.json', "w") as f:
    json.dump(predictions_json, f, indent=4)
    
    
def create_short_answer(entry):
    answer = []    
    for short_answer in entry["short_answers"]:
        if short_answer["start_token"] > -1:
            answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))
    if entry["yes_no_answer"] != "NONE":
        answer.append(entry["yes_no_answer"])
    return " ".join(answer)

def create_long_answer(entry):
    answer = []
    if entry["long_answer"]["start_token"] > -1:
        answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))
    return " ".join(answer)

test_answers_df = pd.read_json("predictions.json")
for var_name in ['long_answer_score','short_answer_score','answer_type']:
    test_answers_df[var_name] = test_answers_df['predictions'].apply(lambda q: q[var_name])
test_answers_df["long_answer"] = test_answers_df["predictions"].apply(create_long_answer)
test_answers_df["short_answer"] = test_answers_df["predictions"].apply(create_short_answer)
test_answers_df["example_id"] = test_answers_df["predictions"].apply(lambda q: str(q["example_id"]))

long_answers = dict(zip(test_answers_df["example_id"], test_answers_df["long_answer"]))
short_answers = dict(zip(test_answers_df["example_id"], test_answers_df["short_answer"]))

class ShowPrediction:
    def __init__(self,jsonl_file):
        self._data = {}
        with jsonlines.open(jsonl_file,'r') as f:
            for d in f:
                self._data[int(d['example_id'])]={
                    'text': d['document_text'],
                    'question': d['question_text']
                }
    def __call__(self,prediction,include_full_text=True):
        data = self._data[prediction['example_id']]
        res = {'question': data['question']}
        if include_full_text:
            res['text'] = data['text']
        for type_ in ['long_answer','short_answers']:
            ans = prediction[type_]
            if isinstance(ans,list):
                ans = ans[0]
            start,end = ans['start_token'],ans['end_token']
            res[type_] = ' '.join(data['text'].split()[start:end])
        return res
    
show_pred = ShowPrediction(nq_test_file)
for pred in np.random.choice(predictions_json['predictions'],1):
        print(json.dumps(show_pred(pred,include_full_text=False),indent=4))
        
        