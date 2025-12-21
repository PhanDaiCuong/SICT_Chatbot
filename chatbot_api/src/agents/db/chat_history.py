import mysql.connector
import logging

logging.basicConfig(level=logging.INFO)

def create_chat_history_table(DB_CONFIG: dict) -> None:
    """
        Creates the `chat_history` table in the MySQL database if it doesn't already exist.
        The table stores chat messages with session ID, message type, and content.
    """
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(255) NOT NULL,
        message_type ENUM('user', 'ai') NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_table_query)
    connection.commit()
    cursor.close()
    connection.close()
    logging.info("Chat history table created (or already exists).")


class MySQLChatMessageHistory:
    """
        A class to manage chat history in MySQL, allowing message storage and retrieval.
    """
    def __init__(self, session_id, DB_CONFIG: dict):
        self.session_id = session_id
        self.DB_CONFIG = DB_CONFIG

    
    def add_message(self, message_type: str, content: str) -> None:
        """
            Adds a message to the database.

            Args:
                message_type (str): Type of the message ('user' or 'ai').
                content (str): The message content.
        """
        connection = mysql.connector.connect(**self.DB_CONFIG)
        cursor = connection.cursor()
        insert_query="""
        INSERT INTO chat_history (session_id, message_type, content)
        VALUES (%s, %s, %s);
        """

        cursor.execute(insert_query, (self.session_id, message_type, content,))
        connection.commit()
        logging.info("Add message successfully!")
        cursor.close()
        connection.close()

    
    def load_messages(self) -> list:
        """
            Retrieves all messages for the session from the database.

            Returns:
                list: List of messages as dictionaries.
        """
        connection = mysql.connector.connect(**self.DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        select_query =  """ SELECT message_type, content 
                            FROM chat_history
                            WHERE session_id = %s ORDER BY created_at;
                        """
        cursor.execute(select_query, (self.session_id,))
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        logging.info("Loading message successfully!")
        
        return [{"type": row["message_type"], "content": row["content"]} for row in rows]
    

    def reset_history(self) -> None:
        """
            Deletes all messages for the session from the database.
        """
        connection = mysql.connector.connect(**self.DB_CONFIG)
        cursor = connection.cursor()
        delete_query = "DELETE FROM chat_history WHERE session_id = %s;"
        cursor.execute(delete_query, (self.session_id,))
        connection.commit()
        logging.info("Reseting message successfully!")
        cursor.close()
        connection.close()

