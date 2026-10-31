"""本家 GraceFM（../gracefm の Python 実装）を import できるようにする。環境変数 GRACEFM_DIR で場所を変えられる。"""
import os
import sys

GRACEFM = os.environ.get("GRACEFM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gracefm"))
sys.path.insert(0, GRACEFM)
