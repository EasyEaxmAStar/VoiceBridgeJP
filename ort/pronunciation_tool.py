"""Inspect, correct/remove entries, or explicitly restore a pronunciation file."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import uuid
from .pronunciation import PronunciationLexicon,_fold

ROOT=Path(__file__).resolve().parents[1]
CANONICAL=ROOT/'config/pronunciation.ja.production.json'
ROLLBACK=ROOT/'config/pronunciation.ja.accepted-two.rollback.json'


def replace_validated(path,content):
    path=Path(path).resolve();lex=PronunciationLexicon(content)
    if path.suffix.casefold()!='.json' or not path.parent.is_dir():raise ValueError('Choose an existing directory and .json destination')
    backup=None
    if path.exists():
        directory=ROOT/'.model-cache/pronunciation-backups';directory.mkdir(parents=True,exist_ok=True)
        backup=directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]+'-'+path.name)
        with backup.open('xb') as handle:handle.write(path.read_bytes())
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as handle:handle.write(content);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    return dict(path=str(path),rule_count=len(lex.entries),sha256=lex.fingerprint,backup=str(backup) if backup else None,
                restart_required=True)


def edit(path,command,term=None,reading=None,source=ROLLBACK):
    path=Path(path)
    if command=='restore':return replace_validated(path,Path(source).read_bytes())
    lex=PronunciationLexicon.load(path)
    if command in {'inspect','validate'}:return lex.metadata
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    key=_fold(term);existing=[r for r in data['entries'] if _fold(r['term'])==key]
    if command=='remove' and not existing:raise ValueError('No exact normalized term to remove')
    data['entries']=[r for r in data['entries'] if _fold(r['term'])!=key]
    if command=='set':data['entries'].append(dict(term=term,reading=reading))
    return replace_validated(path,(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf8'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect','validate','set','remove','restore'])
    parser.add_argument('--path',type=Path,default=CANONICAL)
    parser.add_argument('--term');parser.add_argument('--reading')
    parser.add_argument('--from-file',type=Path,default=ROLLBACK)
    args=parser.parse_args(argv)
    if args.command in {'set','remove'} and not args.term:parser.error('--term is required')
    if args.command=='set' and not args.reading:parser.error('--reading is required')
    try:print(json.dumps(edit(args.path,args.command,args.term,args.reading,args.from_file),ensure_ascii=False,indent=2))
    except (OSError,ValueError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':main()
