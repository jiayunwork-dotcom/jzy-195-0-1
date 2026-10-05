"""应用配置。"""
import os


DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://panel:panel@localhost:5432/panelapp"
)
