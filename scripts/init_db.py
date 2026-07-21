#!/usr/bin/env python3
"""Initialize database tables without running migrations."""
from app.database.session import engine
from app.database.models import Base

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Tables created.")
