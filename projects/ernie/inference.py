# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
import time
__dir__ = os.path.dirname(os.path.abspath(__file__))  #NOLINT
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))  #NOLINT

import numpy as np
import paddle.distributed.fleet as fleet
import paddle.profiler as profiler
from ppfleetx.data.tokenizers import GPTTokenizer
from ppfleetx.core.engine import InferenceEngine, TensorRTConfig
import argparse


class DummyProfiler():
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()

    def start(self):
        pass

    def stop(self):
        pass

    def step(self, *args, **kwargs):
        pass

    def summary(self, *args, **kwargs):
        pass


def parse_args():
    parser = argparse.ArgumentParser(f"ernie inference\n python {sys.argv[0]}")
    parser.add_argument(
        '-m', '--model_dir', type=str, default='./output', help='model dir')
    parser.add_argument(
        '-mp', '--mp_degree', type=int, default=1, help='mp degree')
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        default='',
        choices=['cpu', 'gpu', 'npu', 'intel_gpu'],
        help='device type in [cpu, gpu, npu, intel_gpu]')
    parser.add_argument(
        '--profiling',
        action='store_true',
        default=False,
        help='enable profiling')
    parser.add_argument(
        '--trt', action='store_true', default=False, help='enable profiling')
    parser.add_argument(
        '-b',
        '--batch_size',
        default=int(os.environ.get("BZ", 64)),
        type=int,
        help="batch size")
    parser.add_argument(
        '--dummy',
        default=True,
        action='store_true',
        help='use dummy data for benchmark')
    parser.add_argument(
        '--seed',
        default=1233457890,
        type=int,
        help='random seed for dummy data')
    parser.add_argument(
        '--seqlen', default=384, type=int, help='seqlen for dummy data')
    parser.add_argument(
        '-i',
        '--text',
        type=str,
        default='Hi ERNIE. Tell me who Jack Ma is.',
        help="inputs text")
    args = parser.parse_args()
    return args


def main(args):
    fleet.init(is_collective=True)
    ###########################################################################################################
    # TensorRT inference Engine Config
    # https://github.com/PaddlePaddle/PaddleFleetX/blob/develop/ppfleetx/core/engine/inference_engine.py#L43
    ###########################################################################################################
    if args.trt and (args.device == "gpu"):
        trtc = TensorRTConfig(
            max_batch_size=32,
            workspace_size=1 << 30,
            min_subgraph_size=3,
            precision='fp16',
            use_static=False,
            use_calib_mode=False,
            collect_shape=True,
            shape_range_info_filename=tempfile.NamedTemporaryFile().name)
    else:
        trtc = None
    infer_engine = InferenceEngine(
        args.model_dir,
        args.mp_degree,
        device=args.device,
        tensorrt_config=trtc)

    ###########################################################################################################
    # set batch size in env vars, e.g `export BZ=32`
    ###########################################################################################################
    if args.dummy:
        np.random.seed(args.seed)
        inputs = dict()
        inputs['input_ids'] = np.random.randint(
            40000, size=(args.batch_size, args.seqlen), dtype="int64")
        inputs['token_type_ids'] = np.random.randint(
            4, size=(args.batch_size, args.seqlen), dtype="int64")
        whole_data = [inputs['token_type_ids'], inputs['input_ids']]
    else:
        tokenizer = GPTTokenizer.from_pretrained("gpt2")
        inputs = tokenizer(args.text, padding=True, return_attention_mask=True)
        whole_data = [
            np.array(inputs['token_type_ids']).reshape(1, -1),
            np.array(inputs['input_ids']).reshape(1, -1)
        ]
    if args.profiling:
        prof = profiler.Profiler(
            targets=[
                profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU,
                profiler.ProfilerTarget.Custom
            ],
            scheduler=(3, 10))
    else:
        prof = DummyProfiler()
    with prof:
        for i in range(10):
            start = time.time()
            outs = infer_engine.predict(whole_data)
            paddle.device.synchronize()
            end = time.time()
            cost = f"{(end - start)*1000:.7f}"
            throughput = args.batch_size / (end - start)
            print(
                f"[inference][{i+1}/10]: start: {start:.7f} end:{end:.7f} cost:{cost:>13} ms, throughput: {throughput:.5f} sentence/s"
            )
            prof.step()

    prof.summary(time_unit='ms')


if __name__ == "__main__":
    args = parse_args()
    main(args)
