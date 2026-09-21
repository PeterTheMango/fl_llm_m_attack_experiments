"""Client-owned, durable at-most-once release reservations.

Reserve before private computation. A crash burns the reservation: retries are
refused rather than recomputing independent noise. Counts are conservative upper
bounds on releases, not evidence that a response reached the server.
"""
from contextlib import closing
from pathlib import Path
import sqlite3


class ReleaseLedger:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        with closing(self._connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS clients (client TEXT PRIMARY KEY, budget INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS reservations (client TEXT NOT NULL, request TEXT NOT NULL, PRIMARY KEY (client, request))")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def reserve(self, client, request, budget):
        """Return a local reason or None. Identity and budget come from the client.

        The budget is pinned on first use and cannot be raised by restarting a
        worker. A new policy does not reset this client's accounting scope.
        """
        if not all(isinstance(v, str) and v.strip() for v in (client, request)):
            raise ValueError("client and request must be nonempty strings")
        if type(budget) is not int or budget <= 0:
            raise ValueError("budget must be a positive integer")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO clients VALUES (?, ?)", (client, budget))
            if db.execute("SELECT budget FROM clients WHERE client=?", (client,)).fetchone()[0] != budget:
                raise ValueError("Client release budget changed within accounting scope")
            if db.execute("SELECT 1 FROM reservations WHERE client=? AND request=?", (client, request)).fetchone():
                return "duplicate_request"
            if db.execute("SELECT COUNT(*) FROM reservations WHERE client=?", (client,)).fetchone()[0] >= budget:
                return "budget_exhausted"
            db.execute("INSERT INTO reservations VALUES (?, ?)", (client, request))
        return None

    def count(self, client):
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM reservations WHERE client=?", (client,)).fetchone()[0]
