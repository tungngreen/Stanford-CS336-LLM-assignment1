import logging
import sys # Import sys for StreamHandler

class Logger:
    """
    A simple logger class for single-process applications.
    Configures logging to both a file and the console.
    """
    def __init__(self, name: str = 'my_app_logger', log_file: str = 'app.log', level: int = logging.INFO):
        """
        Initializes the SimpleLogger.

        Args:
            name (str): The name of the logger. Use __name__ in modules.
            log_file (str): The path to the log file.
            level (int): The minimum logging level to handle (e.g., logging.INFO, logging.DEBUG).
                         Messages below this level will be ignored.
        """
        # Get the logger instance by name
        self._logger = logging.getLogger(name)

        # Set the logging level for the logger instance
        self._logger.setLevel(level)

        # Prevent adding duplicate handlers if the logger already exists
        # This is important if you instantiate the class multiple times
        # with the same logger name, or if basicConfig was called previously.
        if not self._logger.handlers:
            # Create a formatter
            formatter = logging.Formatter('[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(funcName)s] - %(message)s')

            # Create a FileHandler to log messages to a file
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)
            except Exception as e:
                print(f"Error setting up file handler for {log_file}: {e}", file=sys.stderr)
                # Continue without file logging if it fails

            # Create a StreamHandler to log messages to the console (stdout)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

            print(f"Logger '{name}' configured successfully.", file=sys.stderr)
        else:
            print(f"Logger '{name}' already has handlers configured. Skipping setup.", file=sys.stderr)


# --- Example Usage ---

if __name__ == "__main__":
    # Instantiate the logger class
    # Logs will go to 'my_single_app.log' and the console at INFO level and above
    app_logger = Logger(name='my_single_app', log_file='my_single_app.log', level=logging.DEBUG)

    print("--- Using SimpleLogger ---")

    # Use the logging methods
    app_logger.debug("This is a debug message.") # Will be logged because level is DEBUG
    app_logger.info("This is an info message.")
    app_logger.warning("This is a warning message.")
    app_logger.error("This is an error message.")
    app_logger.critical("This is a critical message.")

    # Example with an exception
    try:
        result = 1 / 0
    except ZeroDivisionError:
        app_logger.exception("An error occurred during division.")

    print("--- Finished using SimpleLogger ---")
    print("Check 'my_single_app.log' for file output.")

    # Example of getting another logger instance with the same name
    # It will not add duplicate handlers
    another_logger_instance = Logger(name='my_single_app', log_file='another_file.log', level=logging.WARNING)
    another_logger_instance.info("This message goes through the same logger instance.") # Still logs at DEBUG+
    another_logger_instance.warning("This warning message also goes through the same instance.") # Still logs at DEBUG+

    # Example of getting a logger with a different name
    another_named_logger = Logger(name='another_part', log_file='another_part.log', level=logging.WARNING)
    another_named_logger.info("This info message from 'another_part' will NOT be logged (level is WARNING).")
    another_named_logger.warning("This warning message from 'another_part' WILL be logged.")

