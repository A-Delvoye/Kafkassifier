from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
db_path = BASE_DIR / "db.sqlite3"

DATABASE_URL = f"sqlite:///{db_path}"

engine = create_engine(DATABASE_URL, echo=True)

def get_session():
    with Session(engine) as session:
        yield session

def creation():
    SQLModel.metadata.create_all(engine)



