import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from src.db.models import Base
from src.api.app import app, get_db

TEST_DB_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def test_session():
    # StaticPool ensures all sessions share one connection — required for in-memory SQLite
    engine = create_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine)
    session = TestSessionLocal()
    try:
        yield session, TestSessionLocal
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_session):
    session, TestSessionLocal = test_session

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app), TestSessionLocal
    app.dependency_overrides.clear()
