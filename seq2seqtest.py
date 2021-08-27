import logging

import pandas as pd
import torch
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import json

torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

hate_speech = []
counter_narratives = []

f = open("CONAN.json", 'r')
data = json.load(f)

for hs in data['conan']:
	example = []
	example.append(hs['hateSpeech'])
	example.append(hs['counterSpeech'])
	hate_speech.append(example)

print(len(hate_speech))
print(hate_speech[0])

eightyPercent = int(len(hate_speech) * 0.8)
tenPercent = int(len(hate_speech) * 0.1)

train_df = pd.DataFrame(hate_speech[:eightyPercent], columns=["input_text", "target_text"])

eval_df = pd.DataFrame(hate_speech[eightyPercent:eightyPercent + tenPercent], columns=["input_text", "target_text"])

# Configure the model
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 10
#model_args.dataloader_num_workers = 1
#model_args.evaluate_generated_text = True
#model_args.evaluate_during_training = True
#model_args.evaluate_during_training_verbose = False
model_args.fp16 = True
model_args.use_cuda = True

model = Seq2SeqModel(
    "roberta",
    "roberta-base",
    "bert-base-cased",
    args=model_args,
)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
result = model.eval_model(eval_df)

print(result)

predictions = open('predictions_counternarratives', 'w')
for example in hate_speech[eightyPercent + tenPercent:]:
	predictions.write(example[0])
	predictions.write('\n\n')
	predictions.write(str(model.predict([example[0]])))
	predictions.write('\n\n\n\n')

