"""Run on the remote repository to export retained evidence only. No experiments.
Usage: python -m master_script.tools.export_guard_review outputs/guard-v4-pilot NEW_ARCHIVE.tar.gz
Copies SQLite through its read-only backup API; never copies retired weights.
"""
import io,json,sqlite3,sys,tarfile,hashlib
from contextlib import closing
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
state=json.loads((root/'progress.json').read_text())
assert state['active_job'] is None and len(state['completed'])==9
manifest={'files':{},'missing':[]}
with tarfile.open(output,'x:gz') as tar:
 def put(name,data):
  entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o600
  tar.addfile(entry,io.BytesIO(data));manifest['files'][name]=hashlib.sha256(data).hexdigest()
 def add(path):
  if not path.is_file():manifest['missing'].append(str(path));return
  assert not path.is_symlink();put(str(path.relative_to(root)),path.read_bytes())
 for job in state['completed']:
  result=root/job['result'];assert hashlib.sha256(result.read_bytes()).hexdigest()==job['sha256']
  add(result.parent/'manifest.json');artifact=result.parent/'artifacts'/result.stem
  for name in ['result.json','checkpoint-retirement.json']:add(artifact/name)
  for p in artifact.rglob('*.json'):
   if p.name not in ['result.json','checkpoint-retirement.json']:add(p)
  audits=list((artifact/'private-audit').glob('*.jsonl'))
  if not audits:manifest['missing'].append(str(artifact/'private-audit/*.jsonl'))
  for p in audits:add(p)
  ledger=artifact/'client-guard/release-ledger.sqlite'
  if ledger.exists():
   with closing(sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True)) as source, closing(sqlite3.connect(':memory:')) as snapshot:
    source.backup(snapshot);put(str(ledger.relative_to(root)),snapshot.serialize())
  else:manifest['missing'].append(str(ledger))
 # These inputs are named in the frozen configurations. Export their bytes too.
 for rel in ['master_script/configs/guard/approved_training_policy.json','master_script/configs/research_data/squad_rag_study.json']:
  p=Path(rel)
  if p.is_file():put('pinned-inputs/'+p.name,p.read_bytes())
  else:manifest['missing'].append(str(p))
 put('export-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
print(output)
print(hashlib.sha256(output.read_bytes()).hexdigest())
print('Missing:',manifest['missing'])
