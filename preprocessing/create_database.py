from sqlalchemy.orm import sessionmaker
import sys
sys.path.append('.')
sys.path.append('..')
from sqlalchemy import create_engine
from preprocessing.db_orm import Base


if __name__ == '__main__':
    database_path = '/mnt/data/lisa/physionet.org/georgia_100Hz'  #  sys.argv[1]
    engine = create_engine(f'sqlite:////{database_path}/database.db', echo=True)
    Base.metadata.create_all(engine)