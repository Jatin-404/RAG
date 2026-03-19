from app.db.session import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


#FastAPI calls this automatically for every request that needs a DB session. The finally guarantees the session is closed even if the request crashes midway — no connection leaks.