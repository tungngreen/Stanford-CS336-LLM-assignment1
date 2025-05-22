import os
import numpy as np
import regex as re
from collections import Counter
from typing import List, Tuple, Iterable, Iterator
from io import BytesIO
import wandb
import time
import logging

import multiprocessing as mp
import subprocess

from cs336_basics.utils import find_chunk_boundaries, process_chunk,get_pair_stats, \
                               merge_byte_pairs, load_with_pickle, split_chunk
from cs336_basics.trie import BPETrie           

import cProfile
from tqdm import tqdm

def _encode_worker(
    byte_text: bytes,
    vocab_reverse: dict[bytes, int],
    merges: List[tuple[bytes, bytes]],
    split_byte_patterns: re.Pattern,
    encoded_special_tokens: List[bytes],
    
    **kwargs) -> List[int]:
    """
    Encode the input text in bytes using the BPE tokenizer.
    
    Parameters
    ----------
    byte_text : bytes
        The input text to be encoded, in bytes format.

    Returns
    -------
    list[int]
        A list of token IDs representing the encoded text.
    """
    
    def _merge_byte_pairs(pretoken: List[bytes]) -> List[bytes]:
        """
        Merge the byte pairs in the pretoken list according to the merge list.
        
        Parameters
        ----------
        pretoken : List[bytes]
            The input pretoken list to be merged.

        Returns
        -------
        List[bytes]
            A list of merged byte pairs.
        """
        
        for merge in merges:
            i = 0
            merged_result = []
            while i < len(pretoken):
                if i == len(pretoken) - 1:
                    # If we are at the last pretoken, just append it
                    merged_result.append(pretoken[i])
                    i += 1
                    break
                if pretoken[i] == merge[0] and pretoken[i + 1] == merge[1]:
                    # Merge the byte pairs
                    merged_result.append(merge[0] + merge[1])
                    i += 2
                else:
                    merged_result.append(pretoken[i])
                    i += 1
            pretoken = merged_result
            if len(pretoken) == 1:
                # If we have only one pretoken left, we can stop merging
                break
        return pretoken
    
    def _encode_segment(
        segment: bytes
    ) -> List[int]:
        """
        Encode a segment of bytes using simple algorithm that follows the order of merges
        
        Parameters
        ----------
        segment : bytes
            The input segment to be encoded.

        Returns
        -------
        list[int]
            A list of token IDs representing the encoded segment.
        """

        
        logger = logging.getLogger(__name__)
        logger.debug("Encoding segment: {}".format(segment))
        
        # Initialize an empty list to store the encoded result
        encoded_result = []
        
        # Split the segment into pretokens using the GPT2-style regex pattern
        pretokens = split_byte_patterns.finditer(segment)
        
        # Iterate through each pretoken
        for pretoken in pretokens:
            # Check if the pretoken is empty
            if pretoken == b"":
                continue
            
            # Check if the pretoken is in the vocabulary
            if pretoken in vocab_reverse:
                # Append the token ID to the encoded result
                encoded_result.append(vocab_reverse[pretoken])
            else:
                # If the pretoken is not in the vocabulary, we split into bytes
                # and merge the byte pairs
                # Convert the pretoken to a list of bytes
                pretoken_bytes = [bytes([i]) for i in pretoken.group(0)]
                # Merge the byte pairs in the pretoken
                merged_pretoken = _merge_byte_pairs(pretoken_bytes)
                merged_token_ids = []
                for i in range(len(merged_pretoken)):
                    # Check if the merged pretoken is in the vocabulary
                    if merged_pretoken[i] in vocab_reverse:
                        # Append the token ID to the encoded result
                        merged_token_ids.append(vocab_reverse[merged_pretoken[i]])
                    else:
                        # If the merged pretoken is not in the vocabulary, we skip it
                        logger.warning("Un-tokenizable sequence found: {}".format(merged_pretoken[i]))
                encoded_result.extend(merged_token_ids)
                
                # # Check if the merged pretoken is in the vocabulary
                # if merged_pretoken in self.vocab_reverse:
                #     # Append the token ID to the encoded result
                #     encoded_result.append(self.vocab_reverse[merged_pretoken])
                # else:
                #     self.logger.warning("Un-tokenizable sequence found: {}".format(pretoken))
        
        return encoded_result
    
    ########################### Encoder worker function body starts here ##############################
    
    logger = logging.getLogger(__name__)
    
    logger.debug("--- Step 1: Splitting text into pretokens ---")
    # Split the text into pretokens using the GPT2-style regex pattern
    segments = split_chunk(
        chunk=byte_text,
        special_split_tokens=encoded_special_tokens,
        logger=logger
    )
    
    logger.debug("Segments: {}".format(segments))
    
    # Encode the pretokens using the vocabulary
    logger.debug("--- Step 2: Encoding the pretokens ---")
    
    encoded_result = []
    for segment in segments:
        # Check if the segment is a token in the vocabulary
        if segment is None or len(segment) == 0:
            # Skip empty segments
            logger.debug("Skipping empty segment.")
            continue
        # Check if the segment is in the vocabulary
        # which means it is either a single word or a special token
        if segment in vocab_reverse:
            # If yes, we just need to get the token ID using the reverse text-to-ID vocabulary
            encoded_result.append(vocab_reverse[segment])
        else:
            # If no, we need to encode the segment using the BPE algorithm
            # We can use the Trie longest match algorithm
            # to encode the segment
            # Or we can use the simple algorithm that follows the order of merges
            # Trie is faster but it yields different results which we dont know is better
            # So for now we will use the simple algorithm
            encoded_result.extend(
                _encode_segment(segment=segment)
            )

    logger.debug("Encoded result: {}".format(encoded_result))      
    logger.info("--- Encoding completed ---")  
    return encoded_result

def _tokenize_worker(
    worker_id: int,
    vocab_reverse: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    file_path: str,
    start_pos: int,
    end_pos: int,
    split_byte_patterns: re.Pattern,
    special_split_tokens: List[bytes],
    num_buffer_tokens: int = 500_000
):
    """
    Worker function to tokenize a chunk of the text file from `start_pos` to `end_pos`.
    
    Each time, it reads in a block of bytes from the text file object and finds either the last appearance of a split token
    or the last whitespace character before the end of the block.
    It then tokenizes the chunk using the provided vocab and merges.
    This function is designed to be run in parallel across multiple processes.
    It returns a memmaped array of token IDs for the chunk, which will be concatenated together with other chunks later.
    
    Parameters
    ----------
    text_file : bytes
        The input text file in bytes.
    start_pos : int
        The starting offset of the chunk.
    end_pos : int
        The ending offset of the chunk.
    special_split_tokens : List[bytes]
        A list of special tokens to be used for splitting.
        
    Returns
    -------
    List[int]
        A list of paths to the temporary binary files created by the worker.
    """ 
    
    logger = logging.getLogger(__name__)
    
    # Ensure the special_split_tokens is a correct format
    if not isinstance(special_split_tokens, (bytes, list)):
        raise ValueError("special_split_tokens must be a bytes or a list of bytes.")
    if isinstance(special_split_tokens, list):
         if not all(isinstance(token, bytes) for token in special_split_tokens):
              raise TypeError("If special_split_tokens is a list, all elements must be bytes")
         if not special_split_tokens: # Handle empty list case
             raise ValueError("special_split_tokens list is empty. Cannot find delimiters.")
         # If it's a list, keep it as the list for pattern compilation
         split_tokens_list = special_split_tokens
    else:
        # If it's a single bytes object, wrap it in a list for pattern compilation
        split_tokens_list = [special_split_tokens]
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    base_path = f"data/tmp/{base_name}_worker_{worker_id}_"
        
    # Compile a regex pattern to efficiently search for ANY of the specified tokens.
    # Escape each token's bytes and join them with the regex OR operator (|).
    # Example: [b'\n', b'<|end|>'] -> b'\\n|<\\|end\\|>'
    # This pattern will match the first occurrence of any token in the list.
    compiled_split_pattern = re.compile(b"|".join(re.escape(token) for token in split_tokens_list))
    
    token_buffer = []  # Buffer to store the tokens before writing to the file
    partial_file_paths = []
    
    def _flush_buffer_to_new_file():
        """ Flush the current token buffer to a new temporary file. """
        
        if not token_buffer:
            return
        
        part_num = len(partial_file_paths)
        part_path = f"{base_path}_part_{part_num}.bin"
        
        logger.info("Flushing buffer to file: {}".format(part_path))
        np.array(token_buffer, dtype=np.int32).tofile(part_path)
        partial_file_paths.append(part_path)
        token_buffer.clear()  # Clear the buffer after writing
    
    text_file = open(file_path, "rb")
    text_file.seek(start_pos)  # Move to the start position of the worker's chunk
    
    num_bytes_to_process = end_pos - start_pos
    num_bytes_processed = 0
    
    num_total_written_tokens = 0
    output_file_mmap = None
    
    # since 4MB results in about 1 million tokens,
    # we can adjust the size of the blocks to read from the text file so that the number of output tokens
    # will be smaller than the token buffer size (num_buffer_tokens * 4 bytes)
    # This will ensure that we do not run out of memory when processing large files
    # So block size is set to be 1/4 of the buffer size
    # This is a heuristic value that can be adjusted based on the size of the input file and the number of tokens
    block_size = num_buffer_tokens * 4 // 4  # 1/4 of the buffer size in bytes
    
    # Last block read from the text file
    leftover_bytes = b""
    while num_bytes_processed < num_bytes_to_process:
        
        num_bytes_to_read = min(block_size, num_bytes_to_process - num_bytes_processed)
        
        # Read a block of bytes from the text file and add it to the last block
        # If the last block is not empty, it means we have already read some bytes from the last
        # iteration and could not find a split position.
        # Or that after the last split position, we have read some leftover bytes
        # Now we will read the next block and append it to the last block
        # The next block will have the same size as the last block
        new_data = text_file.read(num_bytes_to_read)
        if not new_data:
            # If we reached the end of the file, we can break the loop
            break
        block = leftover_bytes + new_data
        num_bytes_processed += len(new_data)
        
        # Ideal split poitn
        # Find the last occurrence of any of the special split tokens in the block
        last_split_pos = -1
        matches = list(compiled_split_pattern.finditer(block))
        if matches:
            # Find the last match position of the split token
            last_match = matches[-1]
            last_split_pos = last_match.end()  # Get the end position of the last match

        # Fallback 1: If no split token was found, we can find the last line break
        if last_split_pos == -1:
            # Find the last line break in the block
            last_line_break_pos = block.rfind(b"\n")
            # If the last line break is before the end of the block, we can use it as the split position
            # Else we will resort to finding the last whitespace character
            if last_line_break_pos != -1:
                # Use the last line break position as the split position
                last_split_pos = last_line_break_pos + 1
                
        # Fallback 2: If no split token or line break was found, we can find the last whitespace character
        if last_split_pos == -1:
            # Find the last whitespace character in the block
            last_whitespace_pos = block.rfind(b" ")
            if last_whitespace_pos != -1:
                # Use the last whitespace position as the split position
                last_split_pos = last_whitespace_pos + 1
        
        # If we are at the end of the worker's chunk and still cant find a split position,
        # we must process the entire block as a single chunk
        # But this should not happen in practice as the chunk boundaries are designed to
        # ensure that there is always a split position within the block.
        is_final_block = (num_bytes_processed >= num_bytes_to_process)
        # If we cant find a split position and not at the end yet,
        # we will keep the entire block as the last block
        # continue to find the next split position in the next iteration
        if last_split_pos == -1 and not is_final_block:
            leftover_bytes = block  # Keep the entire block as the last block
            continue  # Skip to the next iteration to process the last block
        
        # If we found a split position or its the final block,
        # we process the block up to the last split position
        if last_split_pos != -1:
            chunk_to_process = block[:last_split_pos]
            leftover_bytes = block[last_split_pos:]  # Remaining bytes for the next iteration
        else:
            # If we are at the end of the worker's chunk and still cant find a split position,
            # we must process the entire block as a single chunk
            chunk_to_process = block  # If no split position was found, we process the entire block
            leftover_bytes = b""  # No leftover bytes for the next iteration
        
        if not chunk_to_process:
            continue
        encoded_result = _encode_worker(
            byte_text=chunk_to_process,
            vocab_reverse=vocab_reverse,
            merges=merges,
            split_byte_patterns=split_byte_patterns,
            encoded_special_tokens=special_split_tokens
        )
            
        token_buffer.extend(encoded_result)
        if len(token_buffer) >= num_buffer_tokens:
            _flush_buffer_to_new_file()
            
    # process the leftover bytes if any
    # But there should not be any leftover bytes if the chunk boundaries are set correctly
    # If there are leftover bytes, we need to process them as well
    if leftover_bytes:
        encoded_result = _encode_worker(
            byte_text=leftover_bytes,
            vocab_reverse=vocab_reverse,
            merges=merges,
            split_byte_patterns=split_byte_patterns,
            encoded_special_tokens=special_split_tokens
        )
        
        token_buffer.extend(encoded_result)

    # Flush the remaining tokens in the buffer to a new file
    if token_buffer:
        _flush_buffer_to_new_file()

    logger.info("Worker {} finished processing chunk from {} to {}. Total tokens written: {}".format(
        worker_id, start_pos, end_pos, num_total_written_tokens
    ))
    
    return partial_file_paths

class Tokenizer:
    """
    A base class for tokenizers.
    
    This class is intended to be inherited by specific tokenizer implementations.
    It provides a basic structure and methods that can be extended or overridden
    by subclasses.
    """
    
    def __init__(self):
        True

class BPE_Tokenizer(Tokenizer):
    """
    A class representing a Byte Pair Encoding (BPE) tokenizer.

    This class is designed to handle the training and application of BPE tokenization
    on a given text corpus. It allows for the creation of a vocabulary based on the
    frequency of byte pairs in the text, and provides methods for encoding and decoding
    text using the learned BPE merges.
    """
    
    # GPT2-style regex pattern for splitting the text into potential initial tokens
    split_byte_patterns = br"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
    split_byte_patterns = re.compile(split_byte_patterns)
    
    def __init__(
        self,
        vocab: dict[int, bytes] | str,
        merges: list[tuple[bytes, bytes]] | str,
        special_tokens: list[str] | None = None,
        logger = None,
        verbose: bool = 10, # DEBUG level
    ):
        """
        Initialize the Tokenizer with a vocabulary and merges.
        
        Parameters
        ----------
        vocab : dict[int, bytes]
            A dictionary mapping token IDs to byte strings.
        merges : list[tuple[bytes, bytes]]
            A list of tuples representing the merges.
        special_tokens : list[str] | None
            A list of special tokens
        """
        if logger is None:
            from cs336_basics.logger import Logger
            
            self.logger = Logger(
                name="Tokenizer",
                log_file="tokenizer.log",
                level=verbose,  # INFO level
            )
        self.special_tokens = special_tokens
        if self.special_tokens is not None:
            self.logger.debug("Special tokens: {}".format(self.special_tokens))
        else:
            self.logger.warning("No special tokens provided.")
        
        
        if isinstance(vocab, str) and isinstance(merges, str):
            # If vocab and merges are file paths, load them
            self.from_files(vocab_path=vocab, merges_path=merges, special_tokens=special_tokens)
            
        
        elif isinstance(vocab, dict) and isinstance(merges, list):
            # If vocab is a dictionary and merges is a list, use them directly
            
            # ID to token mapping vocab
            self.vocab = vocab
            # Token to ID mapping vocab
            self.vocab_reverse = {v: k for k, v in vocab.items()}
            self.merges = merges
            
            if len(self.vocab) == 0 or len(self.merges) == 0:
                self.logger.info("Empty tokenizer has been created.")
                
                return
        
            self.vocab_size = len(self.vocab)
            self.merges_size = len(self.merges)
            self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        
            self._handle_special_tokens(special_tokens)
        
            self.logger.info("Tokenizer has been created with a vocabulary of size {} and merges of size {}".format(
                self.vocab_size, self.merges_size
            ))
                
        else:
            raise ValueError("Invalid input types for vocab and merges. Expected str or dict for vocab and list for merges.")
            
        


    def from_files(
        self,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None
    ):
        """
        Load the vocabulary and merges from files.
        
        Parameters
        ----------
        vocab_path : str
            Path to the vocabulary file.
        merges_path : str
            Path to the merges file.
        special_tokens : list[str] | None
            A list of special tokens
        """
        self.vocab, self.merges = load_with_pickle(vocab_path, merges_path)
        
        if special_tokens is not None:
            self.special_tokens = special_tokens

        self.vocab_size = len(self.vocab)
        self.merges_size = len(self.merges)
        
        
        if self.vocab is None or self.merges is None:
            self.logger.warning("Tokenizer loaded with empty vocabulary or merges.")
            return
        
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        self.bpe_trie = BPETrie(self.vocab_reverse)
        
        self._handle_special_tokens(special_tokens)
        
        self.logger.info("Tokenizer loaded with a vocabulary of size {} and merges of size {}".format(
            self.vocab_size, self.merges_size
        ))
        self.logger.debug("Special tokens: {}".format(self.special_tokens))


    def _handle_special_tokens(self, tokens: list[str]):
        """
        Handle special tokens in the input.
        
        Parameters
        ----------
        tokens : list[str]
            A list of strings representing tokens.

        Returns
        -------
        list[bytes]
            A list of byte strings with special tokens handled.
        """
        # Initialize an empty list to store the special tokens converted to bytes.
        # We work with bytes because the training data is read as bytes.
        self.encoded_special_tokens = []
        
        self.special_token_ids = []
        
        if tokens:
            # Encode the tokens to bytes
            self.encoded_special_tokens = [token.encode("utf-8") for token in tokens]
            self.special_token_ids = [self.vocab_reverse[token] for token in self.encoded_special_tokens if token in self.vocab_reverse]

    def prepare_training_data(self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: List[str],
        **kwargs):
        """
        Initializes the BPE tokenizer with a specified vocabulary size and special tokens.

        Parameters
        ----------
        input_path : str | os.PathLike
            Path to the input text file to be used for training the BPE tokenizer.
        vocab_size : int
            The desired size of the vocabulary to be created during training.
        special_tokens : List[str]
            A list of special tokens to be included in the vocabulary.
            These tokens will not be merged with other tokens during training.
            They are typically used for end-of-text markers or other special purposes.  
        """
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.merges: List[tuple[bytes, bytes]] = []
        
        input_path = os.path.abspath(input_path)
        name = input_path.split("/")[-1].split(".")[0].split("_")[0] + "_" + str(vocab_size)
        
        print(kwargs)
        
        if kwargs.get("wandb", None) is not None:
            wandb.init(
                project="bpe-tokenizer",
                name=name,
                entity="local",
                config={
                    "vocab_size": self.vocab_size,
                    "merges_size": self.merges_size,
                    "special_tokens": self.special_tokens
                }
            )
            self.logger.info("WANDB initialized.")
            
        
        # Read the input file in binary mode
        if self.input_path != "":
            with open(input_path, "rb") as f:
                self.file_object = f.read()

        # Make sure text file is not empty
        assert self.file_object != b"", (
            "Text file is empty. Please provide a valid text file."
        )
            
        # --- Step 0: Initial Setup ---
        # Since we are training a BPE tokenizer, our initial vocabulary is 256
        # The special tokens are also added to the vocabulary
        # After each merge, the vocabulary size increases by 1 until it reaches the desired size
        # The vocabulary size is the number of special tokens + 256 base tokens + the number of merges
        self.num_special_tokens = len(special_tokens)
        # Calculate the number of merges needed
        self.num_merges = vocab_size - self.num_special_tokens - 256
        
        # Initialize the original vocabulary of size 256
        self.vocab: dict[int, bytes] = { 
            i: bytes([i]) for i in range(256)
        }
        
        self._handle_special_tokens(special_tokens)
        
    def train(self,
              verbose: bool = False,
              measurement: bool = False,
              parallel: bool = False) -> Tuple[dict[int, bytes], List[tuple[bytes, bytes]]]:
        
        """
        Train the BPE tokenizer on the input text file.
        This method performs the following steps:
        1. Find the chunk boundaries in the text file.
        2. Pretokenize the text file in parallel.
        3. Merge the most frequent pairs of pretokens.
        4. Add special tokens to the vocabulary.
        5. Return the vocabulary and merges.
        Parameters
        ----------
        verbose : bool
            If True, print detailed information about the training process.
        measurement : bool
            If True, enable performance measurement.
        parallel : bool
            If True, use parallel processing for training.
        """

        # --- Step 1: Find the chunk boundaries ---
        self.logger.info("Starting BPE training...")
        # Ideal number of parallel chunks to read
        # It may appear slower for small files because of the overhead of multiprocessing
        # but for large files, it should be faster
        # because it can read multiple chunks in parallel
        # and the overhead of multiprocessing is negligible
        # compared to the time it takes to read the file
        # and process the chunks
        num_chunks = mp.cpu_count() - 1
        
        self.logger.info("--- Step 1: Find the chunk boundaries ---")
        
        
        if measurement:
            profiler_step1 = cProfile.Profile()
            profiler_step1.enable()
            
        start_step = time.time()
        
        # Find chunk boundaries
        chunk_boundaries = find_chunk_boundaries(
            byte_text_file=self.file_object,  # type: ignore
            num_desired_chunks=num_chunks,
            special_split_tokens=self.encoded_special_tokens,
            logger=self.logger
        )
        end_step = time.time()
        
        self.logger.debug("Chunk boundaries found in {} seconds.".format(end_step - start_step))
        if wandb.run is not None:
            wandb.log(
                {
                    "chunk_boundaries": chunk_boundaries,
                    "chunk_boundaries_time": end_step - start_step
                },
                commit=False
            )
        
        
        if measurement:
            profiler_step1.disable()
            print(profiler_step1.print_stats(sort="cumtime"))
            
        
        self.logger.info("--- Step 1: Completed ---")

        # print(f"Chunk boundaries: {chunk_boundaries}")
        # print(text_file[chunk_boundaries[0]:chunk_boundaries[1]])
        
        # --- Step 2: Pretokenize the text file in parallel ---

        self.logger.info("--- Step 2: Pretokenize the text file in parallel ---")
        
        if measurement:
            profiler_step2 = cProfile.Profile()
            profiler_step2.enable()
            
        start_step = time.time()

        
        # This step prepares the raw input text for BPE merging.
        # It typically involves splitting the text into initial "words" or segments.
        # The splitting often respects whitespace and punctuation, and importantly, special tokens.
        # We will count the frequency of these initial segments (pretokens).
        
        # Dictionary to store the frequency of each pretoken
        # Key: The pretoken (bytes)
        # Value: 
        #   - A list of byte objects representing the pretoken. E.g., "hello" as [b'h', b'e', b'l', b'l', b'o'].
        #   - The frequency of the pretoken in the text file.
        # We use bytes as the key because the input text is in bytes
        pretoken_freq: dict[bytes, Tuple[List[bytes], int]] = {}

        
        # Number of chunks to process in parallel
        # it can be less than the number of chunks
        num_processes = len(chunk_boundaries) - 1
        
        # Create a pool of processes
        pool = mp.Pool(processes=num_processes)
        
        # --- Prepare Special Tokens ---
        # Special tokens (like <PAD>, <EOS>) need to be handled explicitly.
        # During pretokenization, we want to ensure these exact sequences are
        # identified and not broken down by the general pretokenization pattern (GPT2_PAT).
        # They are also added to the final vocabulary with dedicated IDs.
            
        # Create tasks for the multiprocessing pool
        # This utilizes the boundaries of the chunks to read the text file in parallel
        # Each task is a tuple containing the arguments for the process_chunk function
        tasks = [
            (self.file_object[chunk_boundaries[i]:chunk_boundaries[i + 1]], 
             self.split_byte_patterns,
             self.encoded_special_tokens,
             self.logger)
            for i in range(num_processes)
        ]
        
        # Aggregate the results from all processes
        # Each process returns a Counter object with the frequency of each pretoken
        # The Counter objects are combined into a single dictionary
        # The frequency of each pretoken is summed across all processes
        aggregated_freq = Counter()
        
        # Use a multiprocessing pool to process the chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            # Map the process_chunk function to the tasks
            results = pool.starmap(process_chunk, tasks)
            
            # Combine the results from all processes
            for result in results:
                aggregated_freq.update(result)

        # Populate the pretoken_freq dictionary with the results
        for pretoken, freq in aggregated_freq.items():
            # Convert the byte objects into a list
            byte_list = [bytes([i]) for i in pretoken]
            # Store the frequency of the pretoken
            pretoken_freq[pretoken] = (byte_list, freq)
            
        end_step = time.time()
        
        self.logger.debug("Pretokenization completed in {} seconds.".format(end_step - start_step))
        if wandb.run is not None:
            wandb.log(
                {
                    "pretokenization_time": end_step - start_step,
                    "num_pretokens": len(pretoken_freq)
                },
                commit=True
            )
    
        if measurement:
            profiler_step2.disable()
            print(profiler_step2.print_stats(sort="cumtime"))


        self.logger.info("--- Step 2: Completed ---")

            # if verbose:
                # # Analyze and print results for Step 2
                # s2 = io.StringIO()
                # sortby2 = 'cumtime' # Sort by cumulative time to see total cost of the function call 
                # # Create a Stats object and print the stats
                # ps2 = pstats.Stats(profiler_step2, stream=s2).sort_stats(sortby2)
                # ps2.print_stats()
                # print(s2.getvalue())
                # print("-" * 30) # Separator
        
        # --- Step 3: Merge the most frequent pairs of pretokens ---

        self.logger.info("--- Step 3: Merge the most frequent pairs of pretokens ---")
        
        if measurement:
            profiler_step3 = cProfile.Profile()
            profiler_step3.enable()
        
        start_step = time.time()
        
        # The merging process is repeated until the desired vocabulary size is reached.
        # The merging process involves finding the most frequent pair of pretokens
        # and merging them into a new pretoken.
        
        # Calculate the initial frequencies of adjacent pairs of pretokens
        byte_pairs_freq, num_byte_tokens_total = get_pair_stats(
            pretoken_freq=pretoken_freq
        )
        
        # if wandb.run is not None:
        #     wandb.log(
        #         {
        #             "num_byte_tokens_total": num_byte_tokens_total,
        #             "iteration": 0
        #         },
        #         commit=True
        #     )
        
        # Loop until the desired vocabulary size is reached
        # The number of merges is equal to the vocabulary size minus the numbers of special tokens and initial tokens
        
        self.logger.debug("Starting BPE merging for {} iterations...".format(self.num_merges))
        for iter in range(self.num_merges):
            start_merge = time.time()
            # Find the best pair to merge: the most frequent pair.
            # max() with a key function finds the item with the maximum value returned by the key function.
            # The key function `lambda pair: (pair_freq[pair], pair)` sorts first by frequency (descending)
            # and then by the pair itself (lexicographically ascending) to break ties consistently.
            best_pair = max(byte_pairs_freq, key=lambda pair: (byte_pairs_freq[pair], pair))
            
            # print(best_pair)
            
            # Making sure the best pair is in an appropriate format
            assert (isinstance(best_pair, tuple) and 
                len(best_pair) == 2 and
                isinstance(best_pair[0], bytes) and
                    isinstance(best_pair[1], bytes)), (
                "Best pair should be a tuple of two bytes. Not {0} and {1}".format(
                    type(best_pair[0]), type(best_pair[1])
                )
            )
            
            # Add the best pair to the merges list
            self.merges.append(best_pair)
            
            # Add the new merged pretoken to the vocabulary
            # The new pretoken is the concatenation of the two bytes in the best pair
            new_pretoken = best_pair[0] + best_pair[1]
            self.vocab[len(self.vocab)] = new_pretoken
            
            # Replace all the occurrences of the best pair in the pretoken_freq dictionary
            # with the new pretoken
            # also update the 'byte_pairs_stat' Counter based on changes.
            
            # If the corpus is too large, parallelize the merging process for efficiency
            num_byte_tokens_total = merge_byte_pairs(
                pretoken_freq=pretoken_freq,
                byte_pairs_freq=byte_pairs_freq,
                best_pair=best_pair,
                logger=self.logger
            )
                
            end_merge = time.time()
            self.logger.debug("Iteration {}: Merging completed in {} seconds.".format(iter + 1, end_merge - start_merge))
            if wandb.run is not None:
                wandb.log(
                    {
                        "merge_time": end_merge - start_merge,
                        "num_byte_tokens_total": num_byte_tokens_total,
                        "iteration": iter + 1
                    }
                )

            # --- Step 4: Add special tokens to the vocabulary ---
            # Add special tokens to the vocabulary
            # The special tokens are added to the vocabulary with dedicated IDs starting from 256 + the number of merges
        
        end_step = time.time()
        self.logger.debug("BPE merging completed in {} seconds.".format(end_step - start_step))
        if wandb.run is not None:
            wandb.log(
                {
                    "bpe_merging_time": end_step - start_step,
                    "num_merges": self.num_merges
                }
            )
        
        if measurement:
            profiler_step3.disable()
            print(profiler_step3.print_stats(sort="cumtime"))
            # if verbose:
            #     # Analyze and print results for Step 3
            #     s3 = io.StringIO()
            #     sortby3 = 'cumtime'
            #     # Create a Stats object and print the stats
            #     ps3 = pstats.Stats(profiler_step3, stream=s3).sort_stats(sortby3)
            #     ps3.print_stats()
            #     print(s3.getvalue())
        
        self.logger.info("--- Step 3: Completed ---")
                
            
        for i in range(self.num_special_tokens):
            # Add the special token to the vocabulary
            self.vocab[len(self.vocab)] = self.encoded_special_tokens[i]
            
        self.logger.info("BPE training completed.")
                
        return self.vocab, self.merges
    
    def encode_segment_trie(self, segment: bytes) -> List[int]:
        """
        Encode a segment of bytes using using Trie longest match.
        
        Parameters
        ----------
        segment : bytes
            The input segment to be encoded.

        Returns
        -------
        list[int]
            A list of token IDs representing the encoded segment.
        """
        
        # Initialize an empty list to store the encoded result
        encoded_result = []
        i = 0
        
        j = 0
        while j < len(segment):
            # Find the longest match in the trie
            token_id, matched_len = self.bpe_trie.longest_match(segment, j)
            
            if matched_len > 0:
                # We found the longest possible match
                # Append the token ID to the encoded result
                encoded_result.append(token_id)
                # Move the index forward by the length of the matched token
                j += matched_len
            else:
                self.logger.warning("Un-tokenizable sequence found at index {}, {}".format(j, segment[j:j+1])) 
                j += 1
                
        return encoded_result
    
    def merge_byte_pairs_for_encode(
        self,
        pretoken: List[bytes],
    ) -> List[bytes]:
        """
        Merge the byte pairs in the pretoken list.
        
        Parameters
        ----------
        pretoken : List[bytes]
            The input pretoken list to be merged.

        Returns
        -------
        List[bytes]
            A list of merged byte pairs.
        """
        
        for merge in self.merges:
            i = 0
            merged_result = []
            while i < len(pretoken):
                if i == len(pretoken) - 1:
                    # If we are at the last pretoken, just append it
                    merged_result.append(pretoken[i])
                    i += 1
                    break
                if pretoken[i] == merge[0] and pretoken[i + 1] == merge[1]:
                    # Merge the byte pairs
                    merged_result.append(merge[0] + merge[1])
                    i += 2
                else:
                    merged_result.append(pretoken[i])
                    i += 1
            pretoken = merged_result
            if len(pretoken) == 1:
                # If we have only one pretoken left, we can stop merging
                break
        return pretoken

    def encode_segment(self, segment: bytes) -> List[int]:
        """
        Encode a segment of bytes using simple algorithm that follows the order of merges
        
        Parameters
        ----------
        segment : bytes
            The input segment to be encoded.

        Returns
        -------
        list[int]
            A list of token IDs representing the encoded segment.
        """
        
        # Initialize an empty list to store the encoded result
        encoded_result = []
        
        # Split the segment into pretokens using the GPT2-style regex pattern
        pretokens = self.split_byte_patterns.finditer(segment)
        
        # Iterate through each pretoken
        for pretoken in pretokens:
            # Check if the pretoken is empty
            if pretoken == b"":
                continue
            
            # Check if the pretoken is in the vocabulary
            if pretoken in self.vocab_reverse:
                # Append the token ID to the encoded result
                encoded_result.append(self.vocab_reverse[pretoken])
            else:
                # If the pretoken is not in the vocabulary, we split into bytes
                # and merge the byte pairs
                # Convert the pretoken to a list of bytes
                pretoken_bytes = [bytes([i]) for i in pretoken.group(0)]
                # Merge the byte pairs in the pretoken
                merged_pretoken = self.merge_byte_pairs_for_encode(pretoken_bytes)
                merged_token_ids = []
                for i in range(len(merged_pretoken)):
                    # Check if the merged pretoken is in the vocabulary
                    if merged_pretoken[i] in self.vocab_reverse:
                        # Append the token ID to the encoded result
                        merged_token_ids.append(self.vocab_reverse[merged_pretoken[i]])
                    else:
                        # If the merged pretoken is not in the vocabulary, we skip it
                        self.logger.warning("Un-tokenizable sequence found: {}".format(merged_pretoken[i]))
                encoded_result.extend(merged_token_ids)
                
                # # Check if the merged pretoken is in the vocabulary
                # if merged_pretoken in self.vocab_reverse:
                #     # Append the token ID to the encoded result
                #     encoded_result.append(self.vocab_reverse[merged_pretoken])
                # else:
                #     self.logger.warning("Un-tokenizable sequence found: {}".format(pretoken))
        
        return encoded_result



    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Encode the input text using the BPE tokenizer.
        
        Parameters
        ----------
        text : str
            The input text to be encoded.

        Returns
        -------
        list[int]
            A list of token IDs representing the encoded text.
        """
        
        self.logger.info("--- Step 0: Encoding the text ---")
        self.logger.debug("Text: {}".format(text))
        # Convert the text to bytes
        byte_text = text.encode("utf-8")
        
        encoded_result = _encode_worker(
            byte_text=byte_text,
            vocab_reverse=self.vocab_reverse,
            merges=self.merges,
            split_byte_patterns=self.split_byte_patterns,
            encoded_special_tokens=self.encoded_special_tokens
        )
        
        return encoded_result
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that
        lazily yields token IDs. This is required for memory-efficient tokenization of large
        files that we cannot directly load into memory.
        
        Parameters
        ----------
        iterable : Iterable[str]
            An iterable of strings to be encoded.

        Returns
        -------
        Iterator[int]
            An iterator that yields token IDs representing the encoded strings.
        """

        for text_chunk in iterable:
            # Encode the text chunk
            encoded_chunk = self.encode(text_chunk)
            # Yield each token ID in the encoded chunk
            self.logger.debug("Yielding token IDs for chunk: {}".format(text_chunk))
            for token_ids in encoded_chunk:
                yield token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into the original text.
        
        Parameters
        ----------
        token_ids : List[int]
            The list of token IDs to be decoded.

        Returns
        -------
        str
            The decoded text.
        """
        
        # Initialize an empty bytearray to store the decoded bytes
        all_decoded_bytes = bytearray() 
        self.logger.debug("Decoding token IDs: {}".format(token_ids))
        
        # Iterate through the token IDs and decode them
        for token_id in token_ids:
            # Check if the token ID is in the vocabulary
            # If it is, append the corresponding bytes to the decoded bytes
            if token_id in self.vocab:
                all_decoded_bytes.extend(self.vocab[token_id]) 
            else:
                # If the token ID is not in the vocabulary, we skip it
                self.logger.warning(f"Un-decodable token ID found: {token_id}")
                # We append a replacement character (U+FFFD) to indicate an error
                all_decoded_bytes.extend(b"\xef\xbf\xbd") # UTF-8 bytes for U+FFFD
        
        # Decode the bytearray to a string
        result = ''
        try:
            result = all_decoded_bytes.decode("utf-8")
            
        except UnicodeDecodeError as e:
            self.logger.warning(f"Malformed final byte sequence after token reconstruction: {e}. Attempting replacement decode.")
            result = all_decoded_bytes.decode("utf-8", errors="replace") # This is the fallback
        self.logger.debug("Decoded result: '{}'".format(result))
        return result

    def tokenize_and_save(
        self,
        input_file: str,
        output_dir: str,
        max_length: int,
        chunk_lines: int = 1000,
        buffer_tokens: int = 50_000_000
    ) -> str:
        """
        Tokenizes a text dataset and saves the tokenized data to a memory-mapped file.
        This focuses on efficiency for large datasets and proper handling
        of tokenization for language models.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Validate inputs (omitted for brevity, assume they are correct based on previous iterations)
        if not os.path.isfile(input_file):
            self.logger.error(f"The input file '{input_file}' does not exist.")
            raise FileNotFoundError(f"The input file '{input_file}' does not exist.")
        if not isinstance(buffer_tokens, int) or buffer_tokens <= 0:
            self.logger.error("`buffer_tokens` must be a positive integer.")
            raise ValueError("`buffer_tokens` must be a positive integer.")
        if not isinstance(max_length, int) or max_length <= 0:
            self.logger.error("`max_length` must be a positive integer.")
            raise ValueError("`max_length` must be a positive integer.")


        base_name = os.path.splitext(os.path.basename(input_file))[0]
        # We'll use a temporary .mmap extension for the intermediate file
        intermediate_mmap_file_path = os.path.join(output_dir, f"{base_name}_intermediate.mmap")
        # The final output will be a proper .npy file
        final_npy_file_path = os.path.join(output_dir, f"{base_name}_tokenized.npy")


        self.logger.info(f"Intermediate mmap file will be at: {intermediate_mmap_file_path}")
        self.logger.info(f"Final .npy file will be saved to: {final_npy_file_path}")

        token_buffer = []
        total_tokens_written = 0
        output_file_mmap = None # This will hold the *current* memmap view

        self.logger.info(f"Starting tokenization of {input_file}...")
        try:
            total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        except Exception:
            total_lines = None

        with open(input_file, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(tqdm(infile, total=total_lines, desc="Tokenizing Lines")):
                line = line.strip()
                if not line:
                    continue

                encoded_output = self.encode(text=line)
                token_ids_for_line = encoded_output

                # DEBUG: Verify the type and values returned by self.encode
                if i == 0: # Only for the first line
                    self.logger.debug(f"First line from encode: {token_ids_for_line[:10]} (first 10)")
                    self.logger.debug(f"Types in first line from encode: {[type(t) for t in token_ids_for_line[:10]]}")

                token_buffer.extend(token_ids_for_line)

                if len(token_buffer) >= buffer_tokens:
                    self.logger.info(f"Buffer reached {len(token_buffer)} tokens, writing to memmap...")
                    num_to_write = len(token_buffer)
                    
                    tokens_to_write_np = np.array(token_buffer, dtype=np.int32)

                    # DEBUG: Verify the type and values of the NumPy array before writing to memmap
                    self.logger.debug(f"Tokens to write NP array dtype: {tokens_to_write_np.dtype}")
                    self.logger.debug(f"Tokens to write NP array first 10: {tokens_to_write_np[:10]}")

                    new_total_size = total_tokens_written + num_to_write

                    if output_file_mmap is not None:
                        output_file_mmap.flush()
                        del output_file_mmap # Close the old mmap view to allow re-opening with new size

                    # Create/re-open the memmap file with the new size
                    output_file_mmap = np.memmap(
                        intermediate_mmap_file_path,
                        dtype=np.int32,
                        mode='w+' if total_tokens_written == 0 else 'r+',
                        shape=(new_total_size,)
                    )

                    output_file_mmap[total_tokens_written : new_total_size] = tokens_to_write_np
                    output_file_mmap.flush()

                    total_tokens_written = new_total_size
                    token_buffer = []

            # Write any remaining tokens in the buffer
            if token_buffer:
                self.logger.info(f"Writing remaining {len(token_buffer)} tokens to memmap...")
                num_to_write = len(token_buffer)
                tokens_to_write_np = np.array(token_buffer, dtype=np.int32)

                # DEBUG: Verify the type and values of the final NumPy array before writing
                self.logger.debug(f"Final Buffer NP array dtype: {tokens_to_write_np.dtype}")
                self.logger.debug(f"Final Buffer NP array first 10: {tokens_to_write_np[:10]}")

                new_total_size = total_tokens_written + num_to_write

                if output_file_mmap is not None:
                    output_file_mmap.flush()
                    del output_file_mmap

                output_file_mmap = np.memmap(
                    intermediate_mmap_file_path,
                    dtype=np.int32,
                    mode='w+' if total_tokens_written == 0 else 'r+',
                    shape=(new_total_size,)
                )

                output_file_mmap[total_tokens_written : new_total_size] = tokens_to_write_np # Fix: use total_tokens_written for offset
                output_file_mmap.flush()
                total_tokens_written = new_total_size
                token_buffer = []

        # At this point, output_file_mmap *might* be None if total_tokens_written was 0 from the start
        # or if the last del output_file_mmap was called.
        # Ensure it's explicitly closed if it exists.
        if output_file_mmap is not None:
            del output_file_mmap

        self.logger.info(f"Tokenization complete. Total tokens written: {total_tokens_written}")
        
        # --- CRITICAL FIX: Re-open the ENTIRE intermediate file for the final np.save ---
        if total_tokens_written > 0:
            self.logger.info(f"Converting intermediate mmap to final .npy file: {final_npy_file_path}")
            
            # This ensures we get a fresh, complete view of the intermediate file.
            # This is the `np.memmap` object that `np.save` will correctly interpret.
            final_mmap_view = np.memmap(
                intermediate_mmap_file_path,
                dtype=np.int32,
                mode='r', # Read-only for saving
                shape=(total_tokens_written,) # Explicitly set the full shape
            )
            
            # DEBUG: Verify data in final_mmap_view before saving
            self.logger.debug(f"Final mmap view dtype before np.save: {final_mmap_view.dtype}")
            self.logger.debug(f"Final mmap view first 10 before np.save: {final_mmap_view[:10]}")

            np.save(final_npy_file_path, final_mmap_view, allow_pickle=False)
            
            # IMPORTANT: Close the final_mmap_view to release the file handle
            del final_mmap_view

            # Clean up the intermediate .mmap file
            os.remove(intermediate_mmap_file_path)
            self.logger.info(f"Final .npy file size: {os.path.getsize(final_npy_file_path) / (1024*1024):.2f} MB")
        else:
            self.logger.warning("No tokens were written, no final .npy file will be created.")
            final_npy_file_path = None

        return final_npy_file_path
    
    def tokenize_and_save_parallel(
        self,
        input_file: str,
        output_dir: str,
        num_processes: int = 1,
        num_buffer_tokens: int = 50_000_000
    ) -> str:
        """
        Tokenizes a text dataset in parallel and saves the tokenized data to a memory-mapped file.
        
        Dividing the input file into chunks (avoiding splitting between tokens) and processing each
        chunk in parallel.
        The result of each process is saved to a memory-mapped file, which is then combined into a 
        final .npy file.
        
        Parameters
        ----------
        input_file : str
            Path to the input text file to be tokenized.
        output_dir : str
            Directory where the tokenized data will be saved.
        max_length : int
            Maximum length of the tokenized sequences.
        num_processes : int, optional
            Number of parallel processes to use for tokenization, by default 1.
        buffer_tokens : int, optional
            Number of tokens to buffer before writing to the memory-mapped file, by default 50_000_000.

        Returns
        -------
        str
            Path to the final .npy file containing the tokenized data.
        """
        
        # Validate inputs (omitted for brevity, assume they are correct based on previous iterations)
        if not os.path.isfile(input_file):
            self.logger.error(f"The input file '{input_file}' does not exist.")
            raise FileNotFoundError(f"The input file '{input_file}' does not exist.")
        if not isinstance(num_buffer_tokens, int) or num_buffer_tokens <= 0:
            self.logger.error("`buffer_tokens` must be a positive integer.")
            raise ValueError("`buffer_tokens` must be a positive integer.")
        if num_processes < 1:
            self.logger.error("`num_processes` must be at least 1.")
            raise ValueError("`num_processes` must be at least 1.")
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        temp_merged_path = os.path.join(output_dir, f"{base_name}_merged_temp.bin")
        final_npy_file_path = os.path.join(output_dir, f"{base_name}_tokenized.npy")

        self.logger.info(f"Final .npy file will be saved to: {final_npy_file_path}")
        
        # Step 1: Open text file and find chunk boundaries
        self.logger.debug(f"Preparing to tokenize {input_file} in parallel with {num_processes} processes...")
        # Open the input file in binary mode to read bytes
        with open(input_file, 'rb') as text_file:
            file_size = os.fstat(text_file.fileno()).st_size
            self.logger.debug(f"Input file size: {file_size} bytes") 

            self.logger.debug(f"Finding chunk boundaries in {input_file}...")
            chunk_boundaries = find_chunk_boundaries(
                byte_text_file=text_file,
                num_desired_chunks=num_processes,
                special_split_tokens=self.encoded_special_tokens
            )
            # Add chunk 0 to the beginning of the file
            chunk_boundaries = [0] + chunk_boundaries
            
            self.logger.debug(f"Found {len(chunk_boundaries)} chunk boundaries: {chunk_boundaries}")
        
        # Step 2: Prepare for parallel tokenization
        # Number of processes is determined by the number of chunk boundaries found
        num_processes = len(chunk_boundaries) - 1
        self.logger.debug(f"Processing {num_processes} chunks in parallel...")
        
        all_partial_files = []        
        # Create tasks for the multiprocessing pool
        if num_processes > 1:
            tasks = [
                (
                    worker_id, self.vocab_reverse, self.merges, input_file,
                    chunk_boundaries[i], chunk_boundaries[i+1],
                    self.split_byte_patterns, self.encoded_special_tokens, num_buffer_tokens
                )
                for i, worker_id in enumerate(range(num_processes))
            ]
            
            all_partial_files = []
            
            with mp.Pool(processes=num_processes) as pool:
                self.logger.debug("Starting parallel tokenization...")
                # Map the process_chunk function to the tasks
                list_of_partials = pool.starmap(_tokenize_worker, tasks)
                
            for chunk_files in list_of_partials:
                all_partial_files.extend(chunk_files)
        elif num_processes == 1:
            # If only one process, we can directly call the worker function
            worker_id = 0
            self.logger.debug("Only one process, tokenizing directly...")
            partial_file = _tokenize_worker(
                worker_id, self.vocab_reverse, self.merges, input_file,
                chunk_boundaries[0], chunk_boundaries[1],
                self.split_byte_patterns, self.encoded_special_tokens, num_buffer_tokens
            )
            if partial_file:
                all_partial_files.extend(partial_file)
        
        if not all_partial_files:
            msg = "No partial files were created during parallel tokenization."
            self.logger.error(msg)
            raise RuntimeError(msg)
        
        self.logger.info(f"All workers completed. Merging {len(all_partial_files)} partial files...")
        
        command = ['cat'] + all_partial_files
        with open(temp_merged_path, 'wb') as merged_file:
            subprocess.run(command, stdout=merged_file, check=True)
        
        self.logger.info(f"Merged {len(all_partial_files)} partial files into {temp_merged_path}.")
        
        # Step 3: Convert the merged binary file to a memory-mapped .npy file
        self.logger.info(f"Converting merged binary file to final .npy file: {final_npy_file_path}")
        
        # Read the merged binary file and convert it to a NumPy array
        with open(temp_merged_path, 'rb') as merged_file:
            token_ids = np.fromfile(merged_file, dtype=np.int32)
        
        # Save the NumPy array to a .npy file
        np.save(final_npy_file_path, token_ids, allow_pickle=False)
        
        # Clean up temporary files
        os.remove(temp_merged_path)
        for partial_file in all_partial_files:
            os.remove(partial_file)
        
        self.logger.info(f"Tokenization complete. Final .npy file saved to: {final_npy_file_path}")
        self.logger.info(f"Final .npy file size: {os.path.getsize(final_npy_file_path) / (1024*1024):.2f} MB")
        
        return final_npy_file_path
        
        
            

if __name__ == "__main__":
    # wandb.init(
    #     project="bpe-tokenizer",
    #     entity="local",
    #     config={
    #         "vocab_size": 1000,
    #         "special_tokens": ["<|endoftext|>"]
    #     },
        
    # )
    kwargs = {
        # "wandb": False,
    }
    if kwargs.get("wandb") is not None:
        wandb.login(
            host="http://wandb-local:8080",
            key="local-457a9e8c8b72f707c6097ca5ed30cf734f3af223"
        )
    
    bpe_tokenizer = BPE_Tokenizer(verbose=10)
    # bpe_tokenizer.prepare_training_data(
    #     input_path="data/TinyStoriesV2-GPT4-valid.txt",
    #     vocab_size=1000,
    #     special_tokens=["<|endoftext|>"],
    #     **kwargs
    # )

    # bpe_tokenizer.train(parallel=False, measurement=False)
    bpe_tokenizer.from_files(
        vocab_path="data/owt_vocab.pkl",
        merges_path="data/owt_merges.pkl",
        special_tokens=["<|endoftext|>"]
    )
    bpe_tokenizer.encode("Hello, world!<|endoftext|>I don't know what I am.<|endoftext|>I just am.")