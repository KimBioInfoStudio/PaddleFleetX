#! /bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

log_dir=log_mp2
rm -rf $log_dir
# export GLOG_v=10

python -m paddle.distributed.launch \
    --devices "2,3" \
    --log_dir=$log_dir projects/ernie/inference.py \
    --mp_degree 2 --model_dir output_6.7B_mp2
