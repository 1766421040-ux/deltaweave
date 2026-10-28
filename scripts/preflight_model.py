"""Offline MPS feasibility probe; fixtures are not benchmark data."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments'/'model-preflight.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    from peft import LoraConfig, get_peft_model
    from audit_tokenizer import encode_prompt, encode_training_example, parse_generated_response, audit
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no CPU/cloud fallback for this probe')
    torch.manual_seed(42)
    model_path = ROOT/'.cache'/'deepseek-r1-distill-qwen-1.5b'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    audit_result = audit(tokenizer, model_path)
    assert audit_result['source_manifest_status'] == 'tokenizer_config_files_match_download_manifest'
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                local_files_only=True, trust_remote_code=False, attn_implementation='sdpa').to('mps')
    torch.mps.synchronize()
    report = {'scope': 'Synthetic feasibility fixtures only; no benchmark or quality claim.',
              'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
              'revision': 'ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562',
              'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
              'load_seconds': time.perf_counter()-start, 'tokenizer_audit': audit_result,
              'generation': [], 'training_steps': []}
    print(json.dumps({'loaded': True, 'seconds': report['load_seconds']}), flush=True)
    class TimeLimit(StoppingCriteria):
        def __init__(self):
            self.start=time.perf_counter(); self.timed_out=False
        def __call__(self, input_ids, scores, **kwargs):
            self.timed_out=time.perf_counter()-self.start>=180
            return self.timed_out
    prompts = [
        'You operate a read-only dependency knowledge assistant. Evidence: snapshot contains alpha==1.0 requiring beta>=2, and beta==2.0 is available. Return precisely one JSON object with action="answer" and arguments={"beta":"2.0"} after reasoning. Do not invent other evidence.',
        'You operate a read-only dependency knowledge assistant. The request asks which release supports the user\'s Python environment, but the Python version is missing. Available actions: ask, answer, abstain. Return one JSON object with action and arguments. Ask for the missing Python version; do not guess.',
    ]
    model.eval()
    for i, prompt in enumerate(prompts):
        ids = encode_prompt(tokenizer, prompt)
        inputs = torch.tensor([ids], device='mps')
        limit = TimeLimit()
        torch.manual_seed(42+i)
        with torch.inference_mode():
            output = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                         max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.95,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                         stopping_criteria=StoppingCriteriaList([limit]), use_cache=True)
        torch.mps.synchronize()
        seconds=time.perf_counter()-limit.start; generated=output[0,len(ids):].tolist()
        ended=bool(generated and generated[-1]==tokenizer.eos_token_id)
        raw=tokenizer.decode(generated[:-1] if ended else generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        try:
            parsed=parse_generated_response(raw); error=None
        except ValueError as exc:
            parsed=None; error=str(exc)
        report['generation'].append({'fixture': i, 'input_tokens':len(ids), 'new_tokens':len(generated),
                   'seconds':seconds, 'tokens_per_second':len(generated)/seconds, 'eos':ended,
                   'timed_out':limit.timed_out,'hit_token_cap':len(generated)>=512 and not ended,
                   'raw':raw,'parsed':parsed,'parse_error':error})
        print(json.dumps({'fixture':i,'tokens':len(generated),'seconds':seconds,'eos':ended}),flush=True)
    model.config.use_cache=False
    model=get_peft_model(model, LoraConfig(r=4,lora_alpha=8,lora_dropout=0.05,bias='none',
             task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    report['trainable_parameters']=sum(p.numel() for p in parameters)
    report['trainable_dtypes']=sorted({str(p.dtype) for p in parameters})
    fixture=encode_training_example(tokenizer,
        'Synthetic compute calibration, not real package evidence.\n'+('entry; '*300)+
        '\nPython version is missing. Return the ask action with field python.',
        {'action':'ask','arguments':{'field':'python'}},'The required environment is absent.',max_length=1024)
    batch={k:torch.tensor([fixture[k]],device='mps') for k in ['input_ids','attention_mask','labels']}
    report['training_sequence_tokens']=len(fixture['input_ids'])
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,foreach=False)
    model.train()
    for step in range(8):
        start=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        loss=model(**batch).loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Non-finite probe loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.0); optimizer.step(); torch.mps.synchronize()
        row={'step':step+1,'loss':float(loss.detach().cpu()),'seconds':time.perf_counter()-start,
             'mps_allocated_bytes':torch.mps.current_allocated_memory(),
             'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory()}
        report['training_steps'].append(row); print(json.dumps(row),flush=True)
    report['host_maxrss_raw']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['host_maxrss_unit']='bytes' if platform.system()=='Darwin' else 'KiB'
    report['memory_note']='MPS allocations sampled after steps, not a proven peak.'
    report['passed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False); stream.write('\n')
    print(json.dumps({'passed':True,'output':str(args.output)}),flush=True)

if __name__=='__main__':
    main()
