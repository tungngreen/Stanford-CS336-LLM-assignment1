import os
import regex as re
from collections import Counter
from typing import List, Tuple, Iterable, Iterator
from io import BytesIO
import wandb
import time

import multiprocessing as mp

from cs336_basics.utils import find_chunk_boundaries, process_chunk,get_pair_stats, \
                               merge_byte_pairs, load_with_pickle, split_chunk
from cs336_basics.trie import BPETrie                               

import cProfile

class BPE_Tokenizer:
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
        vocab: dict[int, bytes] = {},
        merges: list[tuple[bytes, bytes]] = [],
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
        self.special_tokens = special_tokens
        
        # ID to token mapping vocab
        self.vocab = vocab
        # Token to ID mapping vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.vocab_size = len(vocab)
        self.merges_size = len(merges)
    
        
        if logger is None:
            from cs336_basics.logger import Logger
            
            self.logger = Logger(
                name="Tokenizer",
                log_file="tokenizer.log",
                level=verbose,  # INFO level
            )
            
        if len(self.vocab) == 0 or len(self.merges) == 0:
            self.logger._logger.info("Empty tokenizer has been created.")
            
            return
        
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        self.bpe_trie = BPETrie(self.vocab_reverse)
        self.logger._logger.info("Tokenizer has been created with a vocabulary of size {} and merges of size {}".format(
            self.vocab_size, self.merges_size
        ))
        
        if self.special_tokens is not None:
            self.logger._logger.debug("Special tokens: {}".format(self.special_tokens))
        else:
            self.logger._logger.warning("No special tokens provided.")
        self._handle_special_tokens(special_tokens)

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
        self._handle_special_tokens(special_tokens)

        self.vocab_size = len(self.vocab)
        self.merges_size = len(self.merges)
        
        
        if self.vocab is None or self.merges is None:
            self.logger._logger.warning("Tokenizer loaded with empty vocabulary or merges.")
            return
        
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        self.bpe_trie = BPETrie(self.vocab_reverse)
        
        self.logger._logger.info("Tokenizer loaded with a vocabulary of size {} and merges of size {}".format(
            self.vocab_size, self.merges_size
        ))
        self.logger._logger.debug("Special tokens: {}".format(self.special_tokens))


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
            self.logger._logger.info("WANDB initialized.")
            
        
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
        self.logger._logger.info("Starting BPE training...")
        # Ideal number of parallel chunks to read
        # It may appear slower for small files because of the overhead of multiprocessing
        # but for large files, it should be faster
        # because it can read multiple chunks in parallel
        # and the overhead of multiprocessing is negligible
        # compared to the time it takes to read the file
        # and process the chunks
        num_chunks = mp.cpu_count() - 1
        
        self.logger._logger.info("--- Step 1: Find the chunk boundaries ---")
        
        
        if measurement:
            profiler_step1 = cProfile.Profile()
            profiler_step1.enable()
            
        start_step = time.time()
        
        # Find chunk boundaries
        chunk_boundaries = find_chunk_boundaries(
            byte_text_file=self.file_object,  # type: ignore
            num_desired_chunks=num_chunks,
            special_split_tokens=self.encoded_special_tokens,
            logger=self.logger._logger
        )
        end_step = time.time()
        
        self.logger._logger.debug("Chunk boundaries found in {} seconds.".format(end_step - start_step))
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
            
        
        self.logger._logger.info("--- Step 1: Completed ---")

        # print(f"Chunk boundaries: {chunk_boundaries}")
        # print(text_file[chunk_boundaries[0]:chunk_boundaries[1]])
        
        # --- Step 2: Pretokenize the text file in parallel ---

        self.logger._logger.info("--- Step 2: Pretokenize the text file in parallel ---")
        
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
             self.logger._logger)
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
        
        self.logger._logger.debug("Pretokenization completed in {} seconds.".format(end_step - start_step))
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


        self.logger._logger.info("--- Step 2: Completed ---")

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

        self.logger._logger.info("--- Step 3: Merge the most frequent pairs of pretokens ---")
        
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
        
        self.logger._logger.debug("Starting BPE merging for {} iterations...".format(self.num_merges))
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
                logger=self.logger._logger
            )
                
            end_merge = time.time()
            self.logger._logger.debug("Iteration {}: Merging completed in {} seconds.".format(iter + 1, end_merge - start_merge))
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
        self.logger._logger.debug("BPE merging completed in {} seconds.".format(end_step - start_step))
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
        
        self.logger._logger.info("--- Step 3: Completed ---")
                
            
        for i in range(self.num_special_tokens):
            # Add the special token to the vocabulary
            self.vocab[len(self.vocab)] = self.encoded_special_tokens[i]
            
        self.logger._logger.info("BPE training completed.")
                
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
                self.logger._logger.warning("Un-tokenizable sequence found at index {}, {}".format(j, segment[j:j+1])) 
                j += 1
                
        return encoded_result
    
    def merge_byte_pairs(
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
                merged_pretoken = self.merge_byte_pairs(pretoken_bytes)
                merged_token_ids = []
                for i in range(len(merged_pretoken)):
                    # Check if the merged pretoken is in the vocabulary
                    if merged_pretoken[i] in self.vocab_reverse:
                        # Append the token ID to the encoded result
                        merged_token_ids.append(self.vocab_reverse[merged_pretoken[i]])
                    else:
                        # If the merged pretoken is not in the vocabulary, we skip it
                        self.logger._logger.warning("Un-tokenizable sequence found: {}".format(merged_pretoken[i]))
                encoded_result.extend(merged_token_ids)
                
                # # Check if the merged pretoken is in the vocabulary
                # if merged_pretoken in self.vocab_reverse:
                #     # Append the token ID to the encoded result
                #     encoded_result.append(self.vocab_reverse[merged_pretoken])
                # else:
                #     self.logger._logger.warning("Un-tokenizable sequence found: {}".format(pretoken))
        
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
        
        self.logger._logger.info("--- Step 0: Encoding the text ---")
        self.logger._logger.debug("Text: {}".format(text))
        # Convert the text to bytes
        byte_text = text.encode("utf-8")
        
        self.logger._logger.info("--- Step 1: Splitting text into pretokens ---")
        # Split the text into pretokens using the GPT2-style regex pattern
        segments = split_chunk(
            chunk=byte_text,
            special_split_tokens=self.encoded_special_tokens,
            logger=self.logger._logger
        )
        
        self.logger._logger.debug("Segments: {}".format(segments))
        
        # Encode the pretokens using the vocabulary
        self.logger._logger.info("--- Step 2: Encoding the pretokens ---")
        
        encoded_result = []
        for segment in segments:
            # Check if the segment is a token in the vocabulary
            if segment is None or len(segment) == 0:
                # Skip empty segments
                self.logger._logger.debug("Skipping empty segment.")
                continue
            # Check if the segment is in the vocabulary
            # which means it is either a single word or a special token
            if segment in self.vocab_reverse:
                # If yes, we just need to get the token ID using the reverse text-to-ID vocabulary
                encoded_result.append(self.vocab_reverse[segment])
            else:
                # If no, we need to encode the segment using the BPE algorithm
                # We can use the Trie longest match algorithm
                # to encode the segment
                # Or we can use the simple algorithm that follows the order of merges
                # Trie is faster but it yields different results which we dont know is better
                # So for now we will use the simple algorithm
                encoded_result.extend(self.encode_segment(segment))

        self.logger._logger.debug("Encoded result: {}".format(encoded_result))      
        self.logger._logger.info("--- Encoding completed ---")  
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
            self.logger._logger.debug("Yielding token IDs for chunk: {}".format(text_chunk))
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
        self.logger._logger.debug("Decoding token IDs: {}".format(token_ids))
        
        # Iterate through the token IDs and decode them
        for token_id in token_ids:
            # Check if the token ID is in the vocabulary
            # If it is, append the corresponding bytes to the decoded bytes
            if token_id in self.vocab:
                all_decoded_bytes.extend(self.vocab[token_id]) 
            else:
                # If the token ID is not in the vocabulary, we skip it
                self.logger._logger.warning(f"Un-decodable token ID found: {token_id}")
                # We append a replacement character (U+FFFD) to indicate an error
                all_decoded_bytes.extend(b"\xef\xbf\xbd") # UTF-8 bytes for U+FFFD
        
        # Decode the bytearray to a string
        result = ''
        try:
            result = all_decoded_bytes.decode("utf-8")
            
        except UnicodeDecodeError as e:
            self.logger._logger.warning(f"Malformed final byte sequence after token reconstruction: {e}. Attempting replacement decode.")
            result = all_decoded_bytes.decode("utf-8", errors="replace") # This is the fallback
        self.logger._logger.debug("Decoded result: '{}'".format(result))
        return result


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
        vocab_path="data/owt_10k_vocab.pkl",
        merges_path="data/owt_10k_merges.pkl",
        special_tokens=["<|endoftext|>"]
    )
    bpe_tokenizer.encode("Hello, world!<|endoftext|>I don't know what I am.<|endoftext|>I just am.")