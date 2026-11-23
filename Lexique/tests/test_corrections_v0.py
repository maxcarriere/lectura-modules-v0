"""Tests des corrections v0 du 23/11/2026.

- Ouvrir une base SQLite ne la modifie plus (lecture seule, aucun index cree).
- CSV : compter() ne plafonne plus a 1, exporter() ne plante plus,
  filtrer() applique tous les criteres (ou avertit si la colonne manque).
- Base v4/v5 : exporter() garde lemme_id, lemme, cgram et genre.
"""

from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from lectura_lexique import Lexique

TEST_CSV = Path(__file__).parent / "donnees" / "test.csv"
NB_LIGNES_CSV = 34


def _empreinte(chemin: Path) -> str:
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


def _creer_base_v3(chemin: Path, wal: bool = False) -> None:
    conn = sqlite3.connect(str(chemin))
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE formes (ortho TEXT, lemme TEXT, cgram TEXT, phone TEXT, "
        "genre TEXT, nombre TEXT, freq_opensubs REAL, nb_syllabes INTEGER)"
    )
    conn.executemany(
        "INSERT INTO formes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("chat", "chat", "NOM", "ʃa", "m", "s", 45.2, 1),
            ("chats", "chat", "NOM", "ʃa", "m", "p", 12.1, 1),
            ("maison", "maison", "NOM", "mɛzɔ̃", "f", "s", 38.7, 2),
        ],
    )
    conn.commit()
    conn.close()


def _creer_base_v4(chemin: Path) -> None:
    conn = sqlite3.connect(str(chemin))
    conn.executescript("""
        CREATE TABLE formes (
            id INTEGER PRIMARY KEY, ortho TEXT NOT NULL, lemme_id INTEGER,
            multext TEXT, phone TEXT, nb_syllabes INTEGER, syllabes TEXT,
            freq_opensubs REAL
        );
        CREATE TABLE lemmes (
            id INTEGER PRIMARY KEY, lemme TEXT NOT NULL, cgram TEXT NOT NULL,
            genre TEXT, age REAL
        );
        INSERT INTO lemmes VALUES (1, 'chat', 'NOM', 'm', 3.0);
        INSERT INTO lemmes VALUES (2, 'maison', 'NOM', 'f', 4.0);
        INSERT INTO formes VALUES (1, 'chat', 1, 'Ncms', 'ʃa', 1, 'ʃa', 45.2);
        INSERT INTO formes VALUES (2, 'chats', 1, 'Ncmp', 'ʃa', 1, 'ʃa', 12.1);
        INSERT INTO formes VALUES (3, 'maison', 2, 'Ncfs', 'mɛzɔ̃', 2, 'mɛ.zɔ̃', 38.7);
    """)
    conn.commit()
    conn.close()


# ═══ SQLite : ouverture en lecture seule ══════════════════════════════

def test_ouverture_sqlite_ne_modifie_pas_la_base(tmp_path):
    db = tmp_path / "lex.db"
    _creer_base_v3(db)
    avant = _empreinte(db)
    with Lexique(db) as lex:
        assert lex.existe("chat")
        assert lex.phone_de("chat") == "ʃa"
        assert len(lex.formes_de("chat")) == 2
    assert _empreinte(db) == avant
    conn = sqlite3.connect(str(db))
    index = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    conn.close()
    assert index == []


def test_ouverture_sqlite_refuse_toute_ecriture(tmp_path):
    db = tmp_path / "lex.db"
    _creer_base_v3(db)
    with Lexique(db) as lex:
        with pytest.raises(sqlite3.OperationalError):
            lex._get_conn().execute("CREATE TABLE intrus (x)")


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="droits POSIX non root")
def test_base_sqlite_en_lecture_seule(tmp_path):
    """Fichier et dossier en lecture seule, base en mode WAL : l'ouverture marche."""
    dossier = tmp_path / "ro"
    dossier.mkdir()
    db = dossier / "lex.db"
    _creer_base_v3(db, wal=True)
    db.chmod(stat.S_IRUSR)
    dossier.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with Lexique(db) as lex:
            assert lex.existe("maison")
            assert lex.compter(cgram="NOM") == 3
    finally:
        dossier.chmod(stat.S_IRWXU)
        db.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_base_sqlite_absente(tmp_path):
    db = tmp_path / "absente.db"
    with pytest.raises(FileNotFoundError):
        Lexique(db)
    assert not db.exists()


# ═══ CSV : compter, exporter, filtrer ═════════════════════════════════

@pytest.fixture()
def lex_csv():
    with Lexique(TEST_CSV) as lex:
        yield lex


def test_csv_compter_sans_plafond(lex_csv):
    assert lex_csv.compter() == NB_LIGNES_CSV
    assert lex_csv.compter(cgram="NOM") == 7
    assert lex_csv.compter(cgram="VER") == len(lex_csv.filtrer(cgram="VER", limite=0))


def test_csv_filtrer_limite_zero_sans_limite(lex_csv):
    assert len(lex_csv.filtrer(limite=0)) == NB_LIGNES_CSV
    assert len(lex_csv.filtrer(limite=3)) == 3


def test_csv_filtrer_mode_temps_personne(lex_csv):
    res = lex_csv.filtrer(mode="indicatif", temps="imparfait", personne="3", limite=0)
    assert {r["ortho"] for r in res} == {"mangeait"}


def test_csv_filtrer_has_phone(lex_csv):
    avec = lex_csv.compter(has_phone=True)
    sans = lex_csv.compter(has_phone=False)
    assert avec + sans == NB_LIGNES_CSV
    assert avec > 0


def test_csv_critere_sur_colonne_absente_avertit(lex_csv):
    """test.csv n'a pas de colonne age : avertissement, et aucune entree ne passe."""
    with pytest.warns(UserWarning, match="age_min"):
        res = lex_csv.filtrer(age_min=3, limite=0)
    assert res == []
    with pytest.warns(UserWarning, match="has_age"):
        assert lex_csv.compter(has_age=False) == NB_LIGNES_CSV


def test_csv_exporter_csv(lex_csv, tmp_path):
    sortie = tmp_path / "export.csv"
    n = lex_csv.exporter(sortie)
    assert n == NB_LIGNES_CSV
    with open(sortie, encoding="utf-8", newline="") as f:
        lignes = list(csv.DictReader(f))
    assert len(lignes) == NB_LIGNES_CSV
    assert lignes[0]["ortho"] == "chat"
    assert lex_csv.exporter(tmp_path / "noms.csv", cgram="NOM") == 7


def test_csv_exporter_sqlite(lex_csv, tmp_path):
    sortie = tmp_path / "export.db"
    n = lex_csv.exporter(sortie, format="sqlite", cgram="ADJ")
    conn = sqlite3.connect(str(sortie))
    nb = conn.execute("SELECT COUNT(*) FROM formes").fetchone()[0]
    conn.close()
    assert n == nb == lex_csv.compter(cgram="ADJ")


# ═══ Base v4/v5 : exporter garde les colonnes du lemme ════════════════

def test_v4_exporter_csv_garde_lemme_cgram_genre(tmp_path):
    db = tmp_path / "v4.db"
    _creer_base_v4(db)
    sortie = tmp_path / "export.csv"
    with Lexique(db) as lex:
        assert lex.schema_version == 4
        n = lex.exporter(sortie)
    assert n == 3
    with open(sortie, encoding="utf-8", newline="") as f:
        lignes = {r["ortho"]: r for r in csv.DictReader(f)}
    chat = lignes["chat"]
    assert chat["lemme_id"] == "1"
    assert chat["lemme"] == "chat"
    assert chat["cgram"] == "NOM"
    assert chat["genre"] == "m"
    assert lignes["maison"]["genre"] == "f"


def test_v4_exporter_sqlite_et_filtre_age(tmp_path):
    db = tmp_path / "v4.db"
    _creer_base_v4(db)
    sortie = tmp_path / "export.db"
    with Lexique(db) as lex:
        n = lex.exporter(sortie, format="sqlite", age_max=3.5)
    assert n == 2
    conn = sqlite3.connect(str(sortie))
    conn.row_factory = sqlite3.Row
    lignes = [dict(r) for r in conn.execute("SELECT * FROM formes ORDER BY id")]
    conn.close()
    assert [r["ortho"] for r in lignes] == ["chat", "chats"]
    assert all(r["lemme"] == "chat" and r["cgram"] == "NOM" and r["lemme_id"] == 1 for r in lignes)


def test_v4_exporter_critere_non_gere_avertit(tmp_path):
    db = tmp_path / "v4.db"
    _creer_base_v4(db)
    with Lexique(db) as lex:
        with pytest.warns(UserWarning, match="categorie"):
            lex.exporter(tmp_path / "x.csv", categorie="animal")


def test_filtrer_limite_zero_sqlite_sans_limite(tmp_path):
    """limite=0 veut dire « sans limite » sur SQLite comme sur CSV (écart relevé par la documentation du 23/11)."""
    from lectura_lexique import Lexique
    for creer in (_creer_base_v3, _creer_base_v4):
        base = tmp_path / f"{creer.__name__}.db"
        creer(base)
        lex = Lexique(str(base))
        assert len(lex.filtrer(cgram="NOM", limite=0)) == 3, creer.__name__
