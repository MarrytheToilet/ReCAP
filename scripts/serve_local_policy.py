"""Loopback-only OpenAI-compatible endpoint for a local benchmark evaluation."""
import argparse,json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--port',type=int,default=8913);a=p.parse_args()
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200);self.end_headers();self.wfile.write(b'{"status":"ready"}')
        def do_POST(self):
            start=time.monotonic()
            try:
                data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                text=tokenizer.apply_chat_template(data['messages'],tokenize=False,add_generation_prompt=True)
                with lock,torch.inference_mode():
                    batch=tokenizer(text,return_tensors='pt').to('cuda')
                    if batch.input_ids.shape[1]>24000:raise ValueError('prompt exceeds context budget')
                    ids=model.generate(**batch,max_new_tokens=min(data.get('max_tokens',768),1024),do_sample=False,pad_token_id=tokenizer.eos_token_id)
                    output=tokenizer.decode(ids[0,batch.input_ids.shape[1]:],skip_special_tokens=True)
                response=dict(id='local-policy',object='chat.completion',created=int(time.time()),model=a.model,
                    choices=[dict(index=0,message=dict(role='assistant',content=output),finish_reason='stop')],
                    usage=dict(prompt_tokens=int(batch.input_ids.shape[1]),completion_tokens=int(ids.shape[1]-batch.input_ids.shape[1]),total_tokens=int(ids.shape[1])))
                payload=json.dumps(response).encode();self.send_response(200)
                print('request',int(batch.input_ids.shape[1]),response['usage']['completion_tokens'],round(time.monotonic()-start,2),flush=True)
            except Exception as exc:
                payload=json.dumps({'error':{'message':type(exc).__name__,'type':'local_inference_error'}}).encode();self.send_response(500)
            self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)
        def log_message(self,*args):pass
    print('READY',a.port,flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
