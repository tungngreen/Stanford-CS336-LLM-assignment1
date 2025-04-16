import os
import time
from typing import BinaryIO # Import for type hinting binary file objects

def find_chunk_boundaries(
    file: BinaryIO, # The input file object, opened in binary mode.
    desired_num_chunks: int, # The target number of chunks to split the file into.
    split_special_token: bytes # The byte sequence to use as a delimiter for chunk boundaries.
) -> list[int]: # Returns a list of integer file offsets (in bytes) representing the start of each chunk.
    """
    Finds byte offsets in a binary file to serve as chunk boundaries.

    The goal is to split the file into approximately `desired_num_chunks` parts,
    but the actual boundaries are adjusted to coincide with occurrences of the
    `split_special_token`. This ensures that chunks can be processed independently
    without splitting the special token itself.

    May return fewer chunks if the boundaries end up overlapping after adjustment
    or if the file is too short to contain the desired number of token occurrences.

    Args:
        file: A file-like object opened in binary read mode (e.g., open('file.bin', 'rb')).
              The function will seek within this object.
        desired_num_chunks: The preferred number of chunks. The function will aim
                            to place boundaries around file_size / desired_num_chunks intervals.
        split_special_token: The byte sequence that marks valid chunk boundaries.

    Returns:
        A sorted list of unique integer offsets indicating the start position of
        each chunk. The first offset will always be 0, and the last will be the
        total file size.
    """
    # Ensure the split token is provided as bytes, as we're working with a binary file.
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # --- Initial Setup ---
    # Get the total size of the file in bytes.
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    # Reset the file pointer to the beginning.
    file.seek(0)

    # Calculate the approximate size of each chunk based on the desired number.
    # This is just for calculating initial guess positions.
    chunk_size = file_size // desired_num_chunks

    # Generate initial guesses for chunk boundary locations.
    # These are uniformly spaced offsets.
    # The list will contain desired_num_chunks + 1 elements, representing the
    # start of desired_num_chunks segments and the end of the last segment.
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    # Ensure the last boundary is exactly the end of the file.
    chunk_boundaries[-1] = file_size

    # Define a small buffer size for reading chunks of data around the guessed boundaries.
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time to find the token.

    # --- Adjust Boundaries to Token Locations ---
    # Iterate through the *intermediate* boundary guesses (excluding the very start and end).
    # These are the boundaries that need adjustment.
    for bi in range(1, len(chunk_boundaries) - 1):
        # Store the initial guess for the current boundary.
        initial_position = chunk_boundaries[bi]
        # Move the file pointer to the initial guessed position.
        file.seek(initial_position)  # Start scanning from the boundary guess.

        # Search for the split token starting from the initial guess position.
        # We read in small chunks to avoid loading huge files into memory.
        while True:
            # Read a small chunk of data from the current file position.
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk.

            # If we read an empty bytestring, it means we've hit the end of the file.
            # In this case, this boundary should be set to the absolute end of the file.
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break # Stop searching for this boundary.

            # Try to find the special token within the current mini-chunk.
            found_at = mini_chunk.find(split_special_token)
            # If the token was found in the mini-chunk:
            if found_at != -1:
                # Calculate the absolute file offset where the token was found.
                # This is the start position of the mini-chunk + the index within the mini-chunk.
                chunk_boundaries[bi] = initial_position + found_at
                break # We found a suitable boundary, move to the next guessed boundary.
            # If the token was NOT found in the mini-chunk:
            else:
                # Update the initial_position to continue searching in the next mini-chunk.
                # The file pointer is already advanced by file.read().
                initial_position += mini_chunk_size

    # --- Finalize Boundaries ---
    # Convert the list to a set to remove any duplicate boundary positions.
    # This can happen if multiple initial guesses lead to the same token location.
    # Then, convert back to a list and sort the boundaries in ascending order.
    return sorted(set(chunk_boundaries))

## Usage
num_processes = 64  # Number of processes to use for parallelization.
with open("/home/hihi/code/courses/stanford-cs336/Stanford-CS336-LLM-assignment1/data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    start = time.time()
    boundaries = find_chunk_boundaries(
        f, num_processes, "<|endoftext|>".encode("utf-8"))
    end = time.time()
    print(f"Time taken to find chunk boundaries: {end - start:.4f} seconds")
        
    # The following is a serial implementation, but you can parallelize this 
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token