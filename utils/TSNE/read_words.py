def read_words(filepath):
    """
    Read words.txt file, ignoring lines starting with # and extracting
    only the first and last field from each data line.
    
    Args:
        filepath (str): Path to the words.txt file
        
    Returns:
        list of tuples: Each tuple contains (word_id, transcription)
    """
    words = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Split by whitespace
            fields = line.split()
            
            if len(fields) >= 2:
                # First field is word ID, last field is transcription
                word_id = fields[0]
                transcription = fields[-1]
                words.append((word_id, transcription))
    
    return words


if __name__ == "__main__":
    # Example usage
    filepath = "words.txt"
    words = read_words(filepath)
    
    # Display first 10 words
    for word_id, transcription in words[:10]:
        print(f"{word_id}: {transcription}")
    
    print(f"\nTotal words read: {len(words)}")
