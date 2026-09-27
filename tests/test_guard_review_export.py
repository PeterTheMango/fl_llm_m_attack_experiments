"""Retained evidence export works without Python 3.11 SQLite serialization."""
from contextlib import ExitStack, closing
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sqlite3
import sys
import tarfile

import pytest


def test_export_without_serialize_preserves_wal_and_originals(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'master_script/tools/export_guard_review.py'
    pilot = tmp_path / 'pilot'
    archive = tmp_path / 'evidence.tar.gz'
    completed = []
    original_bytes = {}
    connect = sqlite3.connect

    class WithoutSerialize(sqlite3.Connection):
        def __getattribute__(self, name):
            if name == 'serialize':
                raise AttributeError('serialize unavailable on Python 3.10')
            return super().__getattribute__(name)

    with ExitStack() as stack:
        for job in range(9):
            result = pilot / f'results/job-{job:03}/run/0000-result.json'
            artifact = result.parent / 'artifacts' / result.stem
            ledger = artifact / 'client-guard/release-ledger.sqlite'
            ledger.parent.mkdir(parents=True)
            result.write_text('{"status":"complete"}\n')
            completed.append({'job': job, 'result': str(result.relative_to(pilot)),
                              'sha256': sha256(result.read_bytes()).hexdigest()})
            # Keep the writer open so committed data remains in the WAL.
            db = stack.enter_context(closing(connect(ledger)))
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA wal_autocheckpoint=0')
            db.execute('CREATE TABLE reservations (client TEXT, request TEXT)')
            db.execute('INSERT INTO reservations VALUES (?, ?)', ('client0', str(job)))
            db.commit()
            assert Path(str(ledger) + '-wal').stat().st_size > 0
            for path in [result.parent / 'manifest.json', artifact / 'result.json',
                         artifact / 'checkpoint-retirement.json']:
                path.write_text('{}')
            audit = artifact / 'private-audit/answers.jsonl'
            audit.parent.mkdir()
            audit.write_text('{"answer":"fixture"}\n')
            (artifact / 'model.safetensors').write_bytes(b'not exported')
            original_bytes[ledger] = ledger.read_bytes()
            original_bytes[result] = result.read_bytes()
        (pilot / 'progress.json').write_text(json.dumps({
            'active_job': None, 'completed': completed}))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, 'argv', [str(script), str(pilot), str(archive)])
        monkeypatch.setattr(sqlite3, 'connect', lambda *a, **kw: connect(
            *a, factory=WithoutSerialize, **kw))
        runpy.run_path(str(script), run_name='__main__')
        with tarfile.open(archive) as bundle:
            manifest = json.load(bundle.extractfile('export-manifest.json'))
            for name, expected in manifest['files'].items():
                assert sha256(bundle.extractfile(name).read()).hexdigest() == expected
            assert not any(name.endswith('.safetensors') for name in bundle.getnames())
            ledgers = [name for name in bundle.getnames() if name.endswith('.sqlite')]
            assert len(ledgers) == 9
            for job, name in enumerate(ledgers):
                snapshot = tmp_path / f'check-{job}.sqlite'
                snapshot.write_bytes(bundle.extractfile(name).read())
                with closing(connect(snapshot)) as db:
                    assert db.execute('SELECT * FROM reservations').fetchall() == [('client0', str(job))]
        assert all(path.read_bytes() == contents for path, contents in original_bytes.items())
        archive_bytes = archive.read_bytes()
        with pytest.raises(FileExistsError):
            runpy.run_path(str(script), run_name='__main__')
        assert archive.read_bytes() == archive_bytes
