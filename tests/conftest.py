import pytest
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from src.db.models import Base
from src.api.app import app, get_db, get_chroma
from src.vectorstore.chroma_client import ChromaClient

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
def test_chroma():
    # Isolated Chroma per test in a temp directory
    tmp_dir = tempfile.mkdtemp()
    chroma = ChromaClient(persist_dir=tmp_dir)
    yield chroma
    # Note: we don't clean up tmp_dir because Chroma holds file handles on Windows;
    # use ignore_cleanup_errors=True if needed at a higher level


@pytest.fixture(scope="function")
def client(test_session, test_chroma):
    session, TestSessionLocal = test_session
    chroma = test_chroma

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    def override_get_chroma() -> ChromaClient:
        return chroma

    # Inject test instances via dependency overrides
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_chroma] = override_get_chroma

    yield TestClient(app), TestSessionLocal, chroma
    app.dependency_overrides.clear()
