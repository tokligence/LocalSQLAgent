#!/usr/bin/env python3
"""
Centralized logging system for Tokligence LocalSQLAgent
Provides file and console logging with rotation and formatting
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml
import json

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "default_config.yaml"

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class Logger:
    """Centralized logger with file and console output"""

    _instance = None
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize logging configuration"""
        # Load configuration
        self.config = self._load_config()

        # Set up root logger
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)

        # Add handlers
        self._setup_file_handler()
        self._setup_console_handler()

    def _load_config(self) -> dict:
        """Load configuration from file"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    'date_format': '%Y-%m-%d %H:%M:%S',
                    'file': {
                        'enabled': True,
                        'path': 'logs/',
                        'filename': 'localsqlagent.log',
                        'max_bytes': 10485760,
                        'backup_count': 5
                    },
                    'console': {
                        'enabled': True,
                        'colorize': True
                    }
                }
            }

    def _setup_file_handler(self):
        """Set up rotating file handler"""
        if not self.config['logging']['file']['enabled']:
            return

        log_file = LOG_DIR / self.config['logging']['file']['filename']

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config['logging']['file']['max_bytes'],
            backupCount=self.config['logging']['file']['backup_count']
        )

        file_formatter = logging.Formatter(
            fmt=self.config['logging']['format'],
            datefmt=self.config['logging']['date_format']
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, self.config['logging']['level']))

        self.root_logger.addHandler(file_handler)

    def _setup_console_handler(self):
        """Set up console handler with optional color"""
        if not self.config['logging']['console']['enabled']:
            return

        console_handler = logging.StreamHandler(sys.stdout)

        if self.config['logging']['console']['colorize']:
            console_formatter = ColoredFormatter(
                fmt=self.config['logging']['format'],
                datefmt=self.config['logging']['date_format']
            )
        else:
            console_formatter = logging.Formatter(
                fmt=self.config['logging']['format'],
                datefmt=self.config['logging']['date_format']
            )

        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, self.config['logging']['level']))

        self.root_logger.addHandler(console_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a named logger"""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = logger
        return self._loggers[name]

    def log_query(self, query: str, database: str, result: dict):
        """Special logging for SQL queries"""
        query_log_file = LOG_DIR / "queries.log"

        query_entry = {
            "timestamp": datetime.now().isoformat(),
            "database": database,
            "query": query,
            "success": result.get('success', False),
            "execution_time": result.get('execution_time', 0),
            "rows_returned": result.get('row_count', 0),
            "error": result.get('error', None)
        }

        # Write to queries log
        with open(query_log_file, 'a') as f:
            f.write(json.dumps(query_entry) + '\n')

        # Also log to main log
        logger = self.get_logger('query_audit')
        if query_entry['success']:
            logger.info(f"Query executed: {query[:100]}... | DB: {database} | Time: {query_entry['execution_time']:.2f}s | Rows: {query_entry['rows_returned']}")
        else:
            logger.error(f"Query failed: {query[:100]}... | DB: {database} | Error: {query_entry['error']}")

    def log_error(self, component: str, error: Exception, context: dict = None):
        """Special error logging with context"""
        error_log_file = LOG_DIR / "errors.log"

        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }

        # Write to errors log
        with open(error_log_file, 'a') as f:
            f.write(json.dumps(error_entry) + '\n')

        # Also log to main log
        logger = self.get_logger(component)
        logger.error(f"{type(error).__name__}: {str(error)}", exc_info=True)


# Create singleton instance
logger_instance = Logger()


def get_logger(name: str) -> logging.Logger:
    """Get a named logger"""
    return logger_instance.get_logger(name)


def log_query(query: str, database: str, result: dict):
    """Log a query execution"""
    logger_instance.log_query(query, database, result)


def log_error(component: str, error: Exception, context: dict = None):
    """Log an error with context"""
    logger_instance.log_error(component, error, context)