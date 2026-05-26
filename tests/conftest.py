"""pytest 配置 — 将项目根目录加入 sys.path，使 src 可导入。"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
