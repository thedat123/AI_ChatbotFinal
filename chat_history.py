from typing import Dict, List, Any, Optional
import pyodbc
import uuid
import datetime
import json
import logging

class ChatHistoryManager:
    """Class to manage chat history in a SQL database"""
    
    def __init__(self, sql_config: Dict[str, str]):
        """Initialize the chat history manager with SQL connection parameters"""
        self.sql_config = sql_config
        self.logger = logging.getLogger(__name__)
        
        # Create an initial connection to validate configuration
        try:
            conn_str = self._build_connection_string()
            conn = pyodbc.connect(conn_str)
            conn.close()
            self.logger.info("Successfully connected to database")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise
            
        # Create tables if they don't exist
        self._create_tables()
    
    def _build_connection_string(self) -> str:
        """Create a connection string from the SQL configuration"""
        conn_parts = []
        for key, value in self.sql_config.items():
            conn_parts.append(f"{key}={value}")
        return ';'.join(conn_parts)
    
    def _create_tables(self) -> None:
        """Create the necessary tables if they don't exist"""
        conn_str = self._build_connection_string()
        
        try:
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ChatSessions')
                CREATE TABLE ChatSessions (
                    SessionId NVARCHAR(50) PRIMARY KEY,
                    PatientId NVARCHAR(50) NOT NULL,
                    SessionName NVARCHAR(255),
                    CreatedAt DATETIME DEFAULT GETDATE(),
                    UpdatedAt DATETIME DEFAULT GETDATE(),
                    IsActive BIT DEFAULT 1
                )
                """)
                
                # Create messages table
                cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ChatMessages')
                CREATE TABLE ChatMessages (
                    MessageId INT IDENTITY(1,1) PRIMARY KEY,
                    SessionId NVARCHAR(50) NOT NULL,
                    IsFromPatient BIT NOT NULL,
                    MessageText NVARCHAR(MAX) NOT NULL,
                    Context NVARCHAR(MAX),
                    QueryType NVARCHAR(50),
                    ProcessingTime FLOAT,
                    CreatedAt DATETIME DEFAULT GETDATE(),
                    FOREIGN KEY (SessionId) REFERENCES ChatSessions(SessionId)
                )
                """)
                
                conn.commit()
                self.logger.info("Tables created or already exist")
                
        except Exception as e:
            self.logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    def create_session(self, patient_id: str) -> str:
        """Create a new chat session for a patient and return the SessionId"""
        session_id = str(uuid.uuid4())
        conn_str = self._build_connection_string()
        
        try:
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO ChatSessions (SessionId, PatientId, SessionName) VALUES (?, ?, ?)",
                    session_id, patient_id, f"Phiên tư vấn - {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"
                )
                conn.commit()
                
            self.logger.info(f"Created session {session_id} for patient {patient_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create session: {str(e)}")
            return None
    
    def get_active_session(self, patient_id: str) -> str:
        """Get the active session for a patient or create one if none exists"""
        conn_str = self._build_connection_string()
        
        try:
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT SessionId FROM ChatSessions WHERE PatientId = ? AND IsActive = 1 ORDER BY UpdatedAt DESC",
                    patient_id
                )
                
                row = cursor.fetchone()
                if row:
                    session_id = row[0]
                    self.logger.info(f"Found active session {session_id} for patient {patient_id}")
                    return session_id
                else:
                    # Create new session
                    self.logger.info(f"No active session found for patient {patient_id}, creating new one")
                    return self.create_session(patient_id)
                    
        except Exception as e:
            self.logger.error(f"Failed to get active session: {str(e)}")
            return None
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session"""
        conn_str = self._build_connection_string()
        
        try:
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT SessionId, PatientId, SessionName, CreatedAt, UpdatedAt, IsActive 
                       FROM ChatSessions WHERE SessionId = ?""",
                    session_id
                )
                
                row = cursor.fetchone()
                if row:
                    return {
                        'session_id': row[0],
                        'patient_id': row[1],
                        'session_name': row[2],
                        'created_at': row[3].isoformat() if row[3] else None,
                        'updated_at': row[4].isoformat() if row[4] else None,
                        'is_active': bool(row[5])
                    }
                return None
                    
        except Exception as e:
            self.logger.error(f"Failed to get session info: {str(e)}")
            return None
    
    def store_message(self, session_id: str, is_from_patient: bool, message_text: str, 
                      context: List[Dict[str, Any]] = None, query_type: str = None, 
                      processing_time: float = None) -> bool:
        """Store a message in the database"""
        conn_str = self._build_connection_string()
        
        try:
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                
                # Update session UpdatedAt time
                cursor.execute(
                    "UPDATE ChatSessions SET UpdatedAt = GETDATE() WHERE SessionId = ?",
                    session_id
                )
                
                # Insert message
                cursor.execute(
                    """INSERT INTO ChatMessages 
                       (SessionId, IsFromPatient, MessageText, Context, QueryType, ProcessingTime)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    session_id,
                    is_from_patient,
                    message_text,
                    json.dumps(context) if context else None,
                    query_type,
                    processing_time
                )
                
                conn.commit()
                self.logger.info(f"Stored message in session {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store message: {str(e)}")
            return False
    
    def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the message history for a session"""
        conn_str = self._build_connection_string()
        
        try:
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT MessageId, IsFromPatient, MessageText, Context, QueryType, 
                       ProcessingTime, CreatedAt
                       FROM ChatMessages 
                       WHERE SessionId = ?
                       ORDER BY CreatedAt ASC""",
                    session_id
                )
                
                messages = []
                for row in cursor.fetchall():
                    context_data = None
                    if row[3]:  # Context column
                        try:
                            context_data = json.loads(row[3])
                        except:
                            context_data = None
                            
                    messages.append({
                        'message_id': row[0],
                        'is_from_patient': bool(row[1]),
                        'message_text': row[2],
                        'context': context_data,
                        'query_type': row[4],
                        'processing_time': row[5],
                        'created_at': row[6].isoformat() if row[6] else None
                    })
                    
                    if len(messages) >= limit:
                        break
                        
                self.logger.info(f"Retrieved {len(messages)} messages from session {session_id}")
                return messages
                
        except Exception as e:
            self.logger.error(f"Failed to get session history: {str(e)}")
            return []
    
    def format_conversation_history(self, messages: List[Dict[str, Any]]) -> str:
        """Format the conversation history for use in prompts"""
        formatted_history = ""
        
        for msg in messages:
            if msg['is_from_patient']:
                formatted_history += f"Người dùng: {msg['message_text']}\n"
            else:
                formatted_history += f"Trợ lý: {msg['message_text']}\n"
                
        return formatted_history
    
    def get_patient_sessions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a patient"""
        conn_str = self._build_connection_string()
        
        try:
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT SessionId, SessionName, CreatedAt, UpdatedAt, IsActive
                       FROM ChatSessions
                       WHERE PatientId = ?
                       ORDER BY UpdatedAt DESC""",
                    patient_id
                )
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row[0],
                        'session_name': row[1],
                        'created_at': row[2].isoformat() if row[2] else None,
                        'updated_at': row[3].isoformat() if row[3] else None,
                        'is_active': bool(row[4])
                    })
                    
                self.logger.info(f"Retrieved {len(sessions)} sessions for patient {patient_id}")
                return sessions
                
        except Exception as e:
            self.logger.error(f"Failed to get patient sessions: {str(e)}")
            return []
    
    def end_session(self, session_id: str) -> bool:
        """Mark a session as inactive"""
        conn_str = self._build_connection_string()
        
        try:
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE ChatSessions SET IsActive = 0 WHERE SessionId = ?",
                    session_id
                )
                conn.commit()
                
                self.logger.info(f"Ended session {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to end session: {str(e)}")
            return False
    
    def rename_session(self, session_id: str, name: str) -> bool:
        """Rename a session"""
        conn_str = self._build_connection_string()
        
        try:
            with pyodbc.connect(conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE ChatSessions SET SessionName = ? WHERE SessionId = ?",
                    name, session_id
                )
                conn.commit()
                
                self.logger.info(f"Renamed session {session_id} to '{name}'")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to rename session: {str(e)}")
            return False