"""Replay one stateless shared-policy forward, preserving padded Qwen masks exactly."""
from contextlib import ExitStack
from unittest.mock import patch
import types
import torch
from torch import nn
from ..common import require


class GraphPolicy(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model;self.graph=None;self.mode=model.mode
        self.method=model.method;self.small=model.small;self.build_info={}

    def _build(self,batch,commands,features):
        require(not self.model.training,'CUDA graph is inference only')
        self.buffers={k:batch[k].clone() for k in ['sensor','thermal','plc','passport']}
        self.commands=commands.clone();self.features=features.clone()
        def call():return self.model(self.buffers,self.commands,self.features)
        masks=[]
        with torch.inference_mode(),ExitStack() as stack:
            if hasattr(self.model,'backbone'):
                from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
                original=qwen.create_causal_mask
                def remember(*args,**kwargs):
                    value=original(*args,**kwargs);masks.append(value);return value
                with patch.object(qwen,'create_causal_mask',remember):
                    eager,tokens=call()
                require(len(masks)==1,'Unexpected backbone mask path')
                # Captured kernels retain device pointers, not Python ownership.
                # Keep the external mask allocation alive across later replays.
                self.static_masks=masks
                # Fixed padded mask has both 0 and 1; avoid its CUDA->CPU all() branch.
                stack.enter_context(patch.object(qwen,'create_causal_mask',lambda *args,**kwargs:masks[0]))
                modules=[m for m in self.model.modules() if '_update_linear_attn_mask' in type(m).__dict__]
                require(len(modules)==1,'Unsupported Qwen mask layout')
                for module in modules:
                    stack.enter_context(patch.object(module,'_update_linear_attn_mask',
                        types.MethodType(lambda self,mask,past:mask,module)))
            else:eager,tokens=call()
            expected=eager.clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):call()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output,self.tokens=call()
            self.graph.replay()
            torch.testing.assert_close(self.output,expected,atol=1e-5,rtol=1e-5)
            self.build_info={'score_parity_max_error':float((self.output-expected).abs().max()),
                             'tokens':tokens,'padded_tokens':64 if not self.small else tokens,
                             'state_cache':False,'feature_cache':False,'static_masks_preserved':True}

    def forward(self,batch,commands,features):
        require(commands.shape==(1,9,20,4) and features.shape==(1,9,16),'Graph requires one fixed-K decision')
        if self.graph is None:self._build(batch,commands,features)
        with torch.inference_mode():
            for key,value in self.buffers.items():
                require(batch[key].shape==value.shape and batch[key].dtype==value.dtype,'Graph input changed shape/dtype')
                value.copy_(batch[key])
            self.commands.copy_(commands);self.features.copy_(features)
            self.graph.replay()
            return self.output,self.tokens
