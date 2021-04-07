# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data.dataset import Dataset

from .trainer import Trainer
from transformers import Seq2SeqTrainer
from .trainer_utils import PredictionOutput
from .trainer_pt_utils import LabelSmoother
from .utils import logging


if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast


logger = logging.get_logger(__name__)


class ContrastiveSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, set_size=3, contrast_lambda=0.5):
        label_smoother = LabelSmoother(epsilon=0.1)
        target_nll = label_smoother(model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            decoder_input_ids=inputs['target_decoder_input_ids'],
            return_dict=True
        ), inputs['target'])

        pos_ll = torch.exp(-label_smoother(model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            decoder_input_ids=inputs['summary_orig_decoder_input_ids'],
            return_dict=True
        ), inputs['summary_orig']))

        neg_set_ll = torch.exp(torch.stack([
            -label_smoother(model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=inputs[f'summary_contrast_{i}_decoder_input_ids'],
                return_dict=True
            ), inputs[f'summary_contrast_{i}']) for i in range(1, set_size + 1)
        ]))

        contrast_softmax = pos_ll / (pos_ll + neg_set_ll.sum())
        agg_loss = (1 - contrast_lambda) * target_nll - contrast_lambda * -contrast_softmax
        return agg_loss
