import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from app.db.session import engine, init_db, Base
from app.db import models  # noqa — registers models with Base

def main():
    init_db()
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

if __name__ == "__main__":
    main()