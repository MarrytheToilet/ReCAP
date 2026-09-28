"""Check whether tokenizer truncation removes candidate distinctions or answer suffix."""
import argparse,json,hashlib
from pathlib import Path
from transformers import AutoTokenizer
from recap.models.lm_candidate_policy import candidate_prompt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('results/experiments'))
    args=parser.parse_args()
    ROOT=args.data_root
    model=args.model
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    rows=[json.loads(l) for l in (ROOT/'local_controls/shaped_fixed.jsonl').read_text().splitlines()]
    out={}
    for regime,length,obs in [('train',512,360),('deployment',384,220)]:
     counts={'decisions':0,'prompts':0,'truncated_prompts':0,'decisions_with_collision':0,'colliding_candidates':0,'decisions_with_suffix_loss':0};examples=[]
     for row in rows:
      prompts=[candidate_prompt(row,a,max_observation_chars=obs) for a in row['candidates']];encoded=tokenizer(prompts,add_special_tokens=True)['input_ids'];cropped=[tuple(x[:length]) for x in encoded];collision=len(cropped)-len(set(cropped));truncated=sum(len(x)>length for x in encoded)
      counts['decisions']+=1;counts['prompts']+=len(prompts);counts['truncated_prompts']+=truncated;counts['decisions_with_collision']+=bool(collision);counts['colliding_candidates']+=collision;counts['decisions_with_suffix_loss']+=bool(truncated)
      if collision and len(examples)<5:examples.append({'task_id':row.get('task_id'),'step_index':row.get('step_index'),'token_lengths':list(map(len,encoded)),'unique_after_truncation':len(set(cropped)),'candidate_count':len(prompts)})
     out[regime]={'counts':counts,'examples':examples}
    path=ROOT/'prompt_truncation_audit.json';path.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

if __name__ == '__main__':
    main()
