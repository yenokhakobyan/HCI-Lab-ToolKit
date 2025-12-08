"""
Session Manager for Web HCI Collector

Manages participant sessions, tracking state and metadata.
"""

import uuid
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class Session:
    """Represents a participant session."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    is_active: bool = True
    is_calibrated: bool = False
    participant_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    # Data counts
    gaze_samples: int = 0
    face_mesh_samples: int = 0
    emotion_samples: int = 0
    mouse_events: int = 0
    keyboard_events: int = 0

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "is_active": self.is_active,
            "is_calibrated": self.is_calibrated,
            "participant_id": self.participant_id,
            "metadata": self.metadata,
            "gaze_samples": self.gaze_samples,
            "face_mesh_samples": self.face_mesh_samples,
            "emotion_samples": self.emotion_samples,
            "mouse_events": self.mouse_events,
            "keyboard_events": self.keyboard_events,
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
        session_id = str(uuid.uuid4())[:8]
        session = Session(
            session_id=session_id,
            participant_id=participant_id
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

    def end_session(self, session_id: str) -> Optional[Session]:
        """Mark a session as ended."""
        session = self.sessions.get(session_id)
        if session:
            session.is_active = False
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
