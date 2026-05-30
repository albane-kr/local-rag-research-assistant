from sqlalchemy import Column, Integer, String, Text, DateTime, UniqueConstraint, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

Base = declarative_base()


class Resource(Base):
    __tablename__ = "resources"
    __table_args__ = (UniqueConstraint("resource_id", "version", name="uq_resource_version"),)

    id = Column(Integer, primary_key=True)
    resource_id = Column(String, index=True)
    version = Column(Integer)
    raw_path = Column(Text)
    markdown_path = Column(Text, nullable=True)
    status = Column(String, default="ingested")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db(db_url: str = "sqlite:///./data.db"):
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)
