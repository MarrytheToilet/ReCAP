"""Execute explicit public task instructions without oracle paths."""
import argparse,json,re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scripts.eval_pool_baselines import make_env

STOP=set('the a an from of to in on into with within inside floor and that is are be it your you make sure absolutely effort try attempt ensure assure doublecheck look see wide already can then after first step off stop objective once handled following until today all over done this time'.split())
def tokens(text):return set(re.findall(r'[a-z0-9]+',text.lower()))-STOP

def clauses(observation):
    # Only visible introductory task paragraph, never hidden game metadata.
    text=observation.split('-=')[0]
    markers=[text.find(m) for m in ['Here is','Your task',"It's time",'First step','First,','First off','Get ready','Your first'] if m in text]
    if markers:text=text[min(markers):]
    result=[]
    for sentence in re.split(r'[.!?]',text):
        # Conditional subordinate clauses refer to the preceding action.
        sentence=re.sub(r'^\s*(?:After |Once |If |With |Having ).*?,\s*','',sentence,flags=re.I) if ',' in sentence else sentence
        low=sentence.lower()
        nav=re.search(r'\b(?:go|head|venture|travel|move|walk|take a trip)(?: to)?(?: the)? (north|south|east|west)\b',low)
        if nav:result.append(('go',nav.group(1),sentence));continue
        if re.search(r'\bunlock(?:ed)?\b',low):verb='unlock'
        elif re.search(r'\b(?:open(?:ed)?|ajar)\b',low):verb='open'
        elif re.search(r'\b(?:take|retrieve|pick[ -]up|grab|recover|lift)\b',low):verb='take'
        elif re.search(r'\binsert\b',low):verb='insert'
        elif re.search(r'\b(?:put|place|deposit|throw|drop|rest|sit|ditch)\b',low):
            verb='drop' if 'floor' in low else ('insert' if re.search(r'\b(?:into|inside)\b',low) else 'put')
        elif re.search(r'\beat\b',low):verb='eat'
        elif re.search(r'\b(?:close(?:d)?|shut)\b',low):verb='close'
        elif re.search(r'\block(?:ed)?\b',low):verb='lock'
        else:continue
        result.append((verb,None,sentence))
    return result

def choose(clause,admissible):
    verb,direction,text=clause
    if direction:return next((a for a in admissible if a=='go '+direction),None)
    target=tokens(text)-{verb,'unlocked','opened','retrieve','pick','up'}
    options=[a for a in admissible if a.split()[0]==verb]
    if not options:return None
    def score(a):
        words=tokens(a)-{verb};return (len(words&target)/max(len(words),1),len(words&target),-len(words))
    best=max(options,key=score)
    return best if score(best)[1]>0 else None

def episode(task):
    env=make_env(task);history=[];skipped=[]
    try:
        state=env.reset();plan=clauses(state.feedback)
        for clause in plan:
            if len(history)>=30 or state.won:break
            action=choose(clause,state.admissible_commands)
            if action is None:skipped.append(clause);continue
            state,_,done=env.step(action);history.append(action)
            if done:break
        return {'task_id':task,'success':bool(state.won),'plan':plan,'actions':history,'skipped':skipped}
    finally:env.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, default=Path('results/experiments/natural/train.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('results/experiments/public_plan/train.jsonl'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=6)
    args=parser.parse_args()
    tasks=list(dict.fromkeys(json.loads(line)['task_id'] for line in args.tasks.read_text().splitlines()))
    if args.limit is not None: tasks=tasks[:args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as pool: rows=list(pool.map(episode,tasks))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print('tasks',len(rows),'success',sum(r['success'] for r in rows),flush=True)

if __name__=='__main__':main()
