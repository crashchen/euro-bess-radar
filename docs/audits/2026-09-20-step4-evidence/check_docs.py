from pathlib import Path
import json,re,subprocess,unicodedata,os,tempfile
from urllib.parse import unquote,urlsplit
repo=Path.cwd()
tracked=subprocess.check_output(['git','diff','--name-only','fb72dbf'],text=True).splitlines()
paths=[Path(x) for x in tracked if x.endswith('.md')]
paths += sorted(Path('docs/validation').glob('*.md'))
for candidate in ['docs/audits/2026-09-20-step4-handoff.md','docs/audits/2026-09-20-step4-evidence/README.md']:
 if Path(candidate).exists():paths.append(Path(candidate))
errors=[];checked=[]
def slugs(s):
 out=set();used={}
 for line in s.splitlines():
  if not re.match(r'^#{1,6} ',line):continue
  h=re.sub(r'[*`~]','',re.sub(r'^#+\s+','',line)).lower()
  h=''.join(c for c in h if c in '-_ ' or unicodedata.category(c)[0] in 'LN')
  h=h.replace(' ','-');i=used.get(h,0);used[h]=i+1
  out.add(h if i==0 else h+'-'+str(i))
 return out
for p in sorted(set(paths)):
 s=p.read_text();body=re.sub(r'```.*?```','',s,flags=re.S)
 for raw in re.findall(r'\]\(([^\n)]+)\)',body):
  raw=raw.strip('<>');parsed=urlsplit(raw)
  if parsed.scheme or raw.startswith('//'):continue
  dest=p if not parsed.path else p.parent/unquote(parsed.path)
  if not dest.exists():errors.append({'file':str(p),'target':raw,'error':'missing file'})
  elif parsed.fragment and dest.suffix=='.md' and unquote(parsed.fragment) not in slugs(dest.read_text()):errors.append({'file':str(p),'target':raw,'error':'missing heading'})
  checked.append({'file':str(p),'target':raw})
nums=[int(m.group(1)) for m in re.finditer(r'^\| (\d+) \|',Path('docs/runbooks/manual-ui-smoke.md').read_text(),re.M)]
assert nums==list(range(1,48)),nums
print(json.dumps({'files':len(set(paths)),'links_checked':len(checked),'errors':errors,'smoke_numbers':'1–47 sequential'},indent=2))
(Path(os.environ.get('STEP4_OUTPUT',tempfile.gettempdir()))/'doc-link-check.json').write_text(json.dumps({'files':sorted(str(p) for p in set(paths)),'checked':checked,'errors':errors,'smoke_numbers':nums},indent=2)+'\n')
assert not errors
