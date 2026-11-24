"""Correctifs v0 du 24/11/2026.

- Base v4/v5 : ``filtrer()`` et ``compter()`` appliquent genre et nombre meme
  sans ``cgram`` (ils etaient ignores).
- ``iter_sqlite`` ouvre la base en lecture seule et ne cree pas de base vide.
"""

from __future__ import annotations

import sqlite3

import pytest

from lectura_lexique import Lexique
from lectura_lexique._loaders import iter_sqlite
from lectura_lexique._multext import motifs_multext


FORMES = [
    # (id, ortho, lemme_id, multext)
    (1, "chat", 1, "Ncms"),
    (2, "chats", 1, "Ncmp"),
    (3, "maison", 2, "Ncfs"),
    (4, "maisons", 2, "Ncfp"),
    (5, "grande", 3, "Afpfs"),
    (6, "grands", 3, "Afpmp"),
    (7, "mange", 4, "Vmip3s-"),
    (8, "mangent", 4, "Vmip3p-"),
    (9, "sur", 5, "Sp"),
]


@pytest.fixture()
def base_v4(tmp_path):
    chemin = tmp_path / "v4.db"
    conn = sqlite3.connect(str(chemin))
    conn.executescript(
        """
        CREATE TABLE formes (id INTEGER PRIMARY KEY, ortho TEXT, lemme_id INTEGER,
            multext TEXT, phone TEXT, nb_syllabes INTEGER, freq_opensubs REAL,
            freq_composite REAL);
        CREATE TABLE lemmes (id INTEGER PRIMARY KEY, lemme TEXT, cgram TEXT,
            genre TEXT, age REAL);
        """
    )
    conn.executemany(
        "INSERT INTO lemmes VALUES (?, ?, ?, ?, NULL)",
        [(1, "chat", "NOM", "m"), (2, "maison", "NOM", "f"), (3, "grand", "ADJ", None),
         (4, "manger", "VER", None), (5, "sur", "PRE", None)],
    )
    conn.executemany(
        "INSERT INTO formes VALUES (?, ?, ?, ?, '', 1, 1.0, 1.0)", FORMES,
    )
    conn.commit()
    conn.close()
    return chemin


def _orthos(resultats):
    return sorted(r["ortho"] for r in resultats)


def test_motifs_sans_categorie():
    assert motifs_multext() == []
    assert motifs_multext(pos="NOM", genre="feminin") == ["N_f_%"]
    # sans categorie : un motif par categorie qui porte le genre
    motifs = motifs_multext(genre="f")
    assert "N_f_%" in motifs and "A__f_%" in motifs and "V_____f%" in motifs


def test_filtrer_genre_sans_cgram(base_v4):
    lex = Lexique(base_v4)
    assert _orthos(lex.filtrer(genre="f", limite=0)) == ["grande", "maison", "maisons"]
    assert _orthos(lex.filtrer(genre="feminin", limite=0)) == ["grande", "maison", "maisons"]


def test_filtrer_nombre_sans_cgram(base_v4):
    lex = Lexique(base_v4)
    assert _orthos(lex.filtrer(nombre="pluriel", limite=0)) == [
        "chats", "grands", "maisons", "mangent",
    ]


def test_filtrer_genre_et_nombre_sans_cgram(base_v4):
    lex = Lexique(base_v4)
    assert _orthos(lex.filtrer(genre="m", nombre="p", limite=0)) == ["chats", "grands"]


def test_filtrer_avec_cgram_inchange(base_v4):
    lex = Lexique(base_v4)
    assert _orthos(lex.filtrer(cgram="NOM", genre="f", limite=0)) == ["maison", "maisons"]
    assert len(lex.filtrer(limite=0)) == len(FORMES)


def test_compter_genre_nombre_sans_cgram(base_v4):
    lex = Lexique(base_v4)
    assert lex.compter(genre="f") == 3
    assert lex.compter(nombre="p") == 4
    assert lex.compter() == len(FORMES)


def test_iter_sqlite_lecture_seule(tmp_path):
    chemin = tmp_path / "plat.db"
    conn = sqlite3.connect(str(chemin))
    conn.execute("CREATE TABLE formes (ortho TEXT, phon TEXT, freqfilms2 REAL)")
    conn.execute("INSERT INTO formes VALUES ('chat', 'Sa', 45.2)")
    conn.commit()
    conn.close()
    chemin.chmod(0o444)
    try:
        avant = chemin.read_bytes()
        entrees = list(iter_sqlite(chemin))
        assert entrees[0]["ortho"] == "chat"
        assert chemin.read_bytes() == avant
    finally:
        chemin.chmod(0o644)


def test_iter_sqlite_chemin_absent_ne_cree_rien(tmp_path):
    absent = tmp_path / "absente.db"
    with pytest.raises(FileNotFoundError):
        list(iter_sqlite(absent))
    assert not absent.exists()
