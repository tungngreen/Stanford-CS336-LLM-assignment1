class BPETrieNode:
    def __init__(self):
        self.children = {} # Maps byte to another BPETrieNode
        self.token_id = None # Stores the token ID if this node represents a complete token
        self.max_token_len = 0 # Stores the maximum token length in the subtree rooted/starting at this node
        self.path_bytes = b'' # Stores the byte string path from the root to this node. For debugging purposes
        
class BPETrie:
    
    def __init__(self, vocab: dict[bytes, int]):
        self.root = BPETrieNode() # Root node of the trie
        self.root.path_bytes = b'' # Initialize the path bytes of the root node. for debugging purposes
        self._build_trie(vocab)
        
    def _build_trie(self, vocab: dict[bytes, int]):
        
        for token_bytes, token_id in vocab.items():
            current_node = self.root
            
            # Iterate through each byte in the token
            for byte in token_bytes:
                if byte not in current_node.children:
                    # Create a new child node if the byte is not already present
                    new_child = BPETrieNode()
                    new_child.path_bytes = current_node.path_bytes + bytes([byte])
                    current_node.children[byte] = new_child
                # Move down the trie
                current_node = current_node.children[byte]
                # Update the maximum token length in the subtree
                current_node.max_token_len = max(current_node.max_token_len, len(token_bytes))
            
            # Set the token ID at the leaf node
            current_node.token_id = token_id
            # Note: If multiple tokens end at the same node (e.g. if 'A' and 'AB' exist
            # and 'A' is later replaced by 'AB' which shouldn't happen in BPE normally),
            # we might need to store the longest token ID or something similar.
            # For BPE, where merges produce unique longer tokens, the last one wins
            # or the `token_id` is simply set. The greedy search logic will handle it.
        
        return
            
    def longest_match(self, byte_sequence: bytes, start_idx: int) -> tuple[int, int]:
        """
        Find the longest matching token in the trie starting from a given index.
        
        Parameters:
        -----------
        byte_sequence : bytes
            The byte sequence to search in.
        start_idx : int
            The index in the byte sequence to start searching from.

        Returns:
        ----------
        tuple[int, int]
            A tuple containing the token ID of the longest match and its length.        
        """
        current_node = self.root
        longest_match_id = -1
        longest_match_len = 0
        
        # store potential matches as we traverse the trie
        potential_matches = []
        
        for i in range(start_idx, len(byte_sequence)):
            byte = byte_sequence[i]
            if byte in current_node.children:
                current_node = current_node.children[byte]
                current_match_length = i - start_idx + 1
                # If the current node is a complete token
                # it is a potential longest match
                if current_node.token_id is not None:
                    longest_match_id = current_node.token_id
                    longest_match_len = current_match_length
            else:
                break
            
        return longest_match_id, longest_match_len