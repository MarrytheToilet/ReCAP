"""Conservative candidate envelope normalization without rewriting action content."""
import json

def parse_candidates(text):
 text=text.strip()
 if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
 try:obj=json.loads(text)
 except json.JSONDecodeError:
  # Some responses contain an extra object-closing brace after a candidate.
  # Drop only a '}' at array scope, never inside a string or another object.
  stack=[];quoted=False;escaped=False;out=[];removed=0
  for char in text:
   if quoted:
    out.append(char)
    if escaped:escaped=False
    elif char=='\\':escaped=True
    elif char=='"':quoted=False
    continue
   if char=='"':quoted=True
   elif char in '[{':stack.append(char)
   elif char in ']}':
    expected='[' if char==']' else '{'
    if stack and stack[-1]==expected:stack.pop()
    elif char=='}' and stack and stack[-1]=='[':removed+=1;continue
    else:raise ValueError('Ambiguous JSON delimiter mismatch')
   out.append(char)
  if not removed or quoted or stack:raise ValueError('Cannot safely normalize candidate JSON')
  obj=json.loads(''.join(out))
 if isinstance(obj,dict) and 'candidates' not in obj and obj.get('kind') in ['tool','message']:obj={'candidates':[obj]}
 if not isinstance(obj,dict) or not isinstance(obj.get('candidates'),list) or not obj['candidates']:raise ValueError('Missing nonempty candidates array')
 return obj['candidates']
