from datetime import datetime, timezone
from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def now(): return datetime.now(timezone.utc)


class Base(DeclarativeBase): pass


class Essay(Base):
    __tablename__ = 'essays'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    theme: Mapped[str] = mapped_column(Text, default='')
    motivating_text: Mapped[str] = mapped_column(Text, default='')
    status: Mapped[str] = mapped_column(String(32), default='uploaded')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    transcription: Mapped[str] = mapped_column(Text, default='')
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_warnings: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_correction_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    images: Mapped[list['EssayImage']] = relationship(cascade='all, delete-orphan', order_by='EssayImage.position')
    transcriptions: Mapped[list['Transcription']] = relationship(cascade='all, delete-orphan')
    corrections: Mapped[list['Correction']] = relationship(cascade='all, delete-orphan', order_by='Correction.created_at')
    references: Mapped[list['HumanReference']] = relationship(cascade='all, delete-orphan')
    attempts: Mapped[list['AssessmentAttempt']] = relationship(cascade='all, delete-orphan')


class EssayImage(Base):
    __tablename__ = 'images'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    essay_id: Mapped[str] = mapped_column(ForeignKey('essays.id', ondelete='CASCADE'))
    position: Mapped[int] = mapped_column(Integer)
    relative_path: Mapped[str] = mapped_column(Text)


class Transcription(Base):
    __tablename__ = 'transcriptions'
    id: Mapped[int] = mapped_column(primary_key=True)
    essay_id: Mapped[str] = mapped_column(ForeignKey('essays.id', ondelete='CASCADE'))
    text: Mapped[str] = mapped_column(Text)
    confirmed: Mapped[bool] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Correction(Base):
    __tablename__ = 'corrections'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    essay_id: Mapped[str] = mapped_column(ForeignKey('essays.id', ondelete='CASCADE'))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    payload: Mapped[dict] = mapped_column(JSON)
    model_metadata: Mapped[dict] = mapped_column(JSON)
    scores: Mapped[list['CompetencyScore']] = relationship(cascade='all, delete-orphan')
    issues: Mapped[list['Issue']] = relationship(cascade='all, delete-orphan')


class CompetencyScore(Base):
    __tablename__ = 'competency_scores'
    id: Mapped[int] = mapped_column(primary_key=True)
    correction_id: Mapped[str] = mapped_column(ForeignKey('corrections.id', ondelete='CASCADE'))
    competency: Mapped[str] = mapped_column(String(2))
    score: Mapped[int] = mapped_column(Integer)


class Issue(Base):
    __tablename__ = 'issues'
    id: Mapped[int] = mapped_column(primary_key=True)
    correction_id: Mapped[str] = mapped_column(ForeignKey('corrections.id', ondelete='CASCADE'))
    competency: Mapped[str] = mapped_column(String(2))
    payload: Mapped[dict] = mapped_column(JSON)


class HumanReference(Base):
    __tablename__ = 'human_references'
    id: Mapped[int] = mapped_column(primary_key=True)
    essay_id: Mapped[str] = mapped_column(ForeignKey('essays.id', ondelete='CASCADE'))
    correction_id: Mapped[str] = mapped_column(ForeignKey('corrections.id', ondelete='CASCADE'))
    scores: Mapped[dict] = mapped_column(JSON)
    notes: Mapped[str] = mapped_column(Text, default='')
    consent_training: Mapped[bool] = mapped_column(Boolean, default=False)
    partition: Mapped[str] = mapped_column(String(20), default='validation')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AssessmentAttempt(Base):
    __tablename__ = 'assessment_attempts'
    id: Mapped[int] = mapped_column(primary_key=True)
    essay_id: Mapped[str] = mapped_column(ForeignKey('essays.id', ondelete='CASCADE'))
    status: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f'sqlite:///{path}', connect_args={'check_same_thread':False})
    @event.listens_for(engine, 'connect')
    def pragmas(connection, _):
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA busy_timeout=5000')
    Base.metadata.create_all(engine)
    return engine, sessionmaker(engine, expire_on_commit=False)
