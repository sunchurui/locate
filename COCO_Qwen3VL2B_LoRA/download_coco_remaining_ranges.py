import concurrent.futures,hashlib,os,pathlib,threading,time,requests,subprocess
ROOT=pathlib.Path(__file__).resolve().parent/"data/coco2017"; ARCH=ROOT/"archives"
FILES=[("val2017.zip",815585330,"4950dc9d00dbe1c933ee0170f5797584351d2a41"),("annotations_trainval2017.zip",252907541,"8551ee4bb5860311e79dace7e79cb91e432e78b3")]
BASE="https://appcenter-deeplearning.sh1a.qingstor.com/dataset/coco/"; CHUNK=4*1024*1024; WORKERS=int(os.environ.get("COCO_DOWNLOAD_WORKERS", "4")); LOCK=threading.Lock()
def download(name,total,expected):
 out=ARCH/name; md=ARCH/(".ranges_"+name); md.mkdir(exist_ok=True); prefix=out.stat().st_size if out.exists() else 0
 if prefix>total: raise RuntimeError(f"{name}: file too large")
 ranges=[]
 for start in range(prefix,total,CHUNK):
  end=min(total-1,start+CHUNK-1); marker=md/f"{start}_{end}.done"
  if not marker.exists(): ranges.append((start,end,marker))
 fd=os.open(out,os.O_CREAT|os.O_RDWR,0o644); os.ftruncate(fd,total); print(f"{name}: prefix={prefix}, remaining={len(ranges)}, workers={WORKERS}",flush=True); done=[0]; started=time.time()
 def fetch(start,end,marker):
  pos=start; failures=0
  while pos<=end:
   try:
    with requests.get(BASE+name,headers={"Range":f"bytes={pos}-{end}"},stream=True,timeout=(30,120)) as r:
     if r.status_code!=206: raise RuntimeError(f"HTTP {r.status_code}")
     want=f"bytes {pos}-{end}/{total}"
     if r.headers.get("Content-Range")!=want: raise RuntimeError(f"bad Content-Range {r.headers.get('Content-Range')}")
     for block in r.iter_content(1048576):
      if block:
       n=os.pwrite(fd,block,pos)
       if n!=len(block): raise IOError(f"short write {n}/{len(block)}")
       pos+=n
    if pos!=end+1: raise IOError(f"short range {pos-1}")
    marker.touch()
    with LOCK:
     done[0]+=1
     if done[0]%20==0 or done[0]==len(ranges): print(f"{name}: {done[0]}/{len(ranges)} avg {(min(total,prefix+done[0]*CHUNK)/1048576)/max(1,time.time()-started):.2f} MiB/s",flush=True)
    return
   except Exception as e:
    failures+=1; print(f"{name}: retry {start}-{end} at {pos}: {e}",flush=True)
    if failures>=60: raise
    time.sleep(min(5,1+failures//10))
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
   for f in concurrent.futures.as_completed([pool.submit(fetch,a,b,m) for a,b,m in ranges]): f.result()
 finally: os.close(fd)
 h=hashlib.sha1()
 with out.open("rb") as f:
  for block in iter(lambda:f.read(16777216),b""): h.update(block)
 actual=h.hexdigest(); print(f"{name}: sha1={actual}",flush=True)
 if actual!=expected: raise RuntimeError(f"{name}: SHA1 mismatch expected {expected}")
 print(f"{name}: verified",flush=True)
if __name__=="__main__":
 ARCH.mkdir(parents=True,exist_ok=True)
 for item in FILES: download(*item)
 subprocess.run(["unzip","-q","-n",str(ARCH/"val2017.zip"),"-d",str(ROOT)],check=True)
 subprocess.run(["unzip","-q","-n",str(ARCH/"annotations_trainval2017.zip"),"-d",str(ROOT)],check=True)
 n=int(subprocess.check_output(["bash","-lc",f"find {ROOT/'val2017'} -maxdepth 1 -type f -name '*.jpg' | wc -l"]).decode().strip())
 if n!=5000: raise RuntimeError(f"expected 5000 val images, found {n}")
 if not (ROOT/"annotations/instances_val2017.json").is_file(): raise RuntimeError("missing instances_val2017.json")
 print("COCO val2017 image count: 5000; instances_val2017.json present",flush=True)
