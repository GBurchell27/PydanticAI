"""
This is a chat application built using FastAPI and PydanticAI.
It demonstrates how to create an AI-powered chat interface where users can interact with an AI agent.
The app uses SQLite for message storage and provides both synchronous and asynchronous operations.
"""

# Future imports to ensure annotations work correctly across Python versions
from __future__ import annotations as _annotations

# Standard library imports for basic functionality
import asyncio  # For handling asynchronous operations
import json    # For JSON data handling
import sqlite3 # For database operations
from collections.abc import AsyncIterator  # For type hinting async iterators
from concurrent.futures.thread import ThreadPoolExecutor  # For running blocking operations in threads
from contextlib import asynccontextmanager  # For creating async context managers
from dataclasses import dataclass  # For creating data classes with less boilerplate
from datetime import datetime, timezone  # For timestamp handling
from functools import partial  # For creating partial functions
from pathlib import Path  # For handling file paths in a cross-platform way
from typing import Annotated, Any, Callable, Literal, TypeVar  # For type hints

# Third-party imports
import fastapi  # Web framework for building APIs
from fastapi import Depends, Request  # FastAPI components for dependency injection
from fastapi.responses import FileResponse, Response, StreamingResponse  # Different types of HTTP responses
from typing_extensions import LiteralString, ParamSpec, TypedDict  # Additional typing utilities
from dotenv import load_dotenv  # For loading environment variables from .env file
import os

# PydanticAI imports for AI agent functionality
from pydantic_ai import Agent  # Main class for creating AI agents
from pydantic_ai.exceptions import UnexpectedModelBehavior  # Error handling
from pydantic_ai.messages import (  # Message-related components
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

# Load environment variables from .env file (like API keys)
load_dotenv()
print("API Key loaded:", bool(os.getenv('OPENAI_API_KEY')))

# Create an AI agent using OpenAI's GPT-4 model
agent = Agent('openai:gpt-4')
THIS_DIR = Path(__file__).parent  # Get the directory where this script is located


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    This ensures proper database connection handling during the app's lifetime.
    """
    async with Database.connect() as db:
        yield {'db': db}


# Create the FastAPI application instance with the lifespan manager
app = fastapi.FastAPI(lifespan=lifespan)


@app.get('/')
async def index() -> FileResponse:
    """
    Route handler for the root URL ('/').
    Returns the HTML file containing the chat interface.
    """
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')


@app.get('/chat_app.ts')
async def main_ts() -> FileResponse:
    """
    Route handler for serving the TypeScript file.
    The TypeScript code is compiled in the browser for simplicity.
    """
    return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')


async def get_db(request: Request) -> Database:
    """
    Dependency function to get the database instance from the request state.
    Used with FastAPI's dependency injection system.
    """
    return request.state.db


@app.get('/chat/')
async def get_chat(database: Database = Depends(get_db)) -> Response:
    """
    Route handler for getting chat messages.
    Returns all messages from the database as newline-delimited JSON.
    """
    msgs = await database.get_messages()
    return Response(
        b'\n'.join(json.dumps(to_chat_message(m)).encode('utf-8') for m in msgs),
        media_type='text/plain',
    )


class ChatMessage(TypedDict):
    """
    TypedDict defining the structure of chat messages sent to the browser.
    Contains role (user/model), timestamp, and content.
    """
    role: Literal['user', 'model']  # Can only be 'user' or 'model'
    timestamp: str                  # ISO format timestamp
    content: str                    # Message content


def to_chat_message(m: ModelMessage) -> ChatMessage:
    """
    Converts a PydanticAI ModelMessage to our ChatMessage format.
    Handles both user messages (ModelRequest) and AI responses (ModelResponse).
    """
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


@app.post('/chat/')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    """
    Route handler for posting new chat messages.
    Takes a user prompt, processes it with the AI agent, and streams the response.
    """
    async def stream_messages():
        """
        Generator function that streams messages to the client.
        Handles both immediate user message display and AI response streaming.
        """
        # First, send back the user's message immediately
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                }
            ).encode('utf-8')
            + b'\n'
        )
        # Get existing chat history
        messages = await database.get_messages()
        # Process the prompt with the AI agent, streaming the response
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream(debounce_by=0.01):
                # Convert the streamed text to a proper message format
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

        # Save the complete conversation to the database
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')


# Type variables for generic function signatures
P = ParamSpec('P')
R = TypeVar('R')


@dataclass
class Database:
    """
    Database class for managing chat messages in SQLite.
    Uses a thread pool executor to handle SQLite's synchronous operations in an async context.
    """
    con: sqlite3.Connection          # SQLite connection
    _loop: asyncio.AbstractEventLoop # Event loop for async operations
    _executor: ThreadPoolExecutor    # Thread pool for running SQLite operations

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
    ) -> AsyncIterator[Database]:
        """
        Creates and manages a database connection.
        Uses a context manager to ensure proper cleanup of resources.
        """
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        con = await loop.run_in_executor(executor, cls._connect, file)
        slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        """
        Creates the SQLite connection and initializes the messages table if it doesn't exist.
        """
        con = sqlite3.connect(str(file))
        cur = con.cursor()
        cur.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        """
        Adds new messages to the database asynchronously.
        """
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        """
        Retrieves all messages from the database asynchronously.
        Converts the stored JSON data back into ModelMessage objects.
        """
        c = await self._asyncify(
            self._execute, 'SELECT message_list FROM messages order by id'
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        """
        Executes an SQL query with optional commit.
        Uses LiteralString type to prevent SQL injection.
        """
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        """
        Converts synchronous functions to asynchronous by running them in the thread pool.
        This is necessary because SQLite operations are blocking.
        """
        return await self._loop.run_in_executor(
            self._executor,
            partial(func, **kwargs),
            *args,
        )


# Run the application using uvicorn if this file is run directly
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'chat_app:app',
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable auto-reload during development
        reload_dirs=[str(THIS_DIR)]  # Watch this directory for changes
    )
