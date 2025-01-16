def to_ordinal(n: int) -> str:
    """Convert an integer to its ordinal string representation."""
    if not isinstance(n, int):
        raise TypeError("Expected an integer")
        
    if n < 0:
        raise ValueError("Expected a non-negative integer")
        
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
        
    return f"{n}{suffix}"