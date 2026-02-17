"""
Session Manager for Web HCI Collector

Manages participant sessions, tracking state and metadata.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List
from dataclasses import dataclass, field


class SessionStatus(str, Enum):
    """Session lifecycle states."""
    CREATED = "created"           # Session created by researcher, waiting for participant
    CALIBRATING = "calibrating"   # Participant is performing eye calibration
    COLLECTING = "collecting"     # Active data collection (content viewing)
    ANSWERING = "answering"       # Participant is answering questions
    COMPLETED = "completed"       # Session finished successfully
    ABANDONED = "abandoned"       # Participant disconnected without completing


# Valid state transitions
_VALID_TRANSITIONS = {
    SessionStatus.CREATED: {SessionStatus.CALIBRATING, SessionStatus.COLLECTING, SessionStatus.ABANDONED},
    SessionStatus.CALIBRATING: {SessionStatus.COLLECTING, SessionStatus.ABANDONED},
    SessionStatus.COLLECTING: {SessionStatus.ANSWERING, SessionStatus.COMPLETED, SessionStatus.ABANDONED},
    SessionStatus.ANSWERING: {SessionStatus.COMPLETED, SessionStatus.ABANDONED},
    SessionStatus.COMPLETED: set(),
    SessionStatus.ABANDONED: set(),
}


@dataclass
class Session:
    """Represents a participant session."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    status: SessionStatus = SessionStatus.CREATED
    is_calibrated: bool = False
    participant_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    # Experiment configuration
    experiment_config: Dict = field(default_factory=dict)
    # Expected shape: {
    #   "content_url": "/static/experiments/green_energy_v2.1.html",
    #   "experiment_name": "Green Energy Study",
    #   "require_calibration": True,
    # }

    # Data counts
    gaze_samples: int = 0
    face_mesh_samples: int = 0
    emotion_samples: int = 0
    mouse_events: int = 0
    keyboard_events: int = 0
    answer_count: int = 0
    hover_events: int = 0

    @property
    def is_active(self) -> bool:
        """Backward-compatible active check."""
        return self.status in (
            SessionStatus.CREATED,
            SessionStatus.CALIBRATING,
            SessionStatus.COLLECTING,
            SessionStatus.ANSWERING,
        )

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "status": self.status.value,
            "is_active": self.is_active,
            "is_calibrated": self.is_calibrated,
            "participant_id": self.participant_id,
            "metadata": self.metadata,
            "experiment_config": self.experiment_config,
            "gaze_samples": self.gaze_samples,
            "face_mesh_samples": self.face_mesh_samples,
            "emotion_samples": self.emotion_samples,
            "mouse_events": self.mouse_events,
            "keyboard_events": self.keyboard_events,
            "answer_count": self.answer_count,
            "hover_events": self.hover_events,
        }


class SessionManager:
    """
    Manages participant sessions.

    Handles session creation, tracking, and cleanup.
    """

    def __init__(self):
        self.sessions: Dict[str, Session] = {}

    def create_session(self, participant_id: Optional[str] = None) -> Session:
        """Create a new session."""
        session_id = self._unique_id()
        session = Session(
            session_id=session_id,
            participant_id=participant_id,
        )
        self.sessions[session_id] = session
        return session

    def create_session_with_config(
        self,
        participant_id: Optional[str] = None,
        experiment_config: Optional[Dict] = None,
    ) -> Session:
        """Create a new session with experiment configuration."""
        session_id = self._unique_id()
        session = Session(
            session_id=session_id,
            participant_id=participant_id,
            experiment_config=experiment_config or {},
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> Session:
        """Get existing session or create a new one."""
        if session_id in self.sessions:
            return self.sessions[session_id]

        session = Session(session_id=session_id)
        self.sessions[session_id] = session
        return session

    def transition_session(self, session_id: str, new_status: SessionStatus) -> Optional[Session]:
        """
        Transition a session to a new status.

        Returns the session if transition is valid, None otherwise.
        """
        session = self.sessions.get(session_id)
        if not session:
            return None

        valid_next = _VALID_TRANSITIONS.get(session.status, set())
        if new_status not in valid_next:
            print(f"Invalid session transition: {session.status.value} -> {new_status.value}")
            return None

        session.status = new_status
        if new_status in (SessionStatus.COMPLETED, SessionStatus.ABANDONED):
            session.ended_at = datetime.now()
        if new_status == SessionStatus.COLLECTING and not session.is_calibrated:
            session.is_calibrated = True
        return session

    def end_session(self, session_id: str) -> Optional[Session]:
        """Mark a session as ended."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        if session.status in (SessionStatus.COMPLETED, SessionStatus.ABANDONED):
            return session

        # If not completed normally, mark as abandoned
        if session.status != SessionStatus.COMPLETED:
            session.status = SessionStatus.ABANDONED
        session.ended_at = datetime.now()
        return session

    def get_active_sessions(self) -> List[Session]:
        """Get all active sessions."""
        return [s for s in self.sessions.values() if s.is_active]

    def get_all_sessions(self) -> List[Session]:
        """Get all sessions."""
        return list(self.sessions.values())

    def update_session_counts(self, session_id: str, data_type: str, count: int = 1):
        """Update sample counts for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return

        if data_type == "gaze":
            session.gaze_samples += count
        elif data_type == "face_mesh":
            session.face_mesh_samples += count
        elif data_type == "emotion":
            session.emotion_samples += count
        elif data_type == "mouse":
            session.mouse_events += count
        elif data_type == "keyboard":
            session.keyboard_events += count
        elif data_type == "answer":
            session.answer_count += count
        elif data_type == "hover":
            session.hover_events += count

    def set_calibrated(self, session_id: str, calibrated: bool = True):
        """Mark session as calibrated."""
        session = self.sessions.get(session_id)
        if session:
            session.is_calibrated = calibrated

    def set_metadata(self, session_id: str, key: str, value):
        """Set metadata for a session."""
        session = self.sessions.get(session_id)
        if session:
            session.metadata[key] = value

    def _unique_id(self) -> str:
        """Generate a unique 8-character session ID."""
        for _ in range(10):
            candidate = str(uuid.uuid4())[:8]
            if candidate not in self.sessions:
                return candidate
        return str(uuid.uuid4())[:12]
