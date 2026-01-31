from enum import Enum

class TokenType(Enum):
    # Existing token types...
    DRIP = 'DRIP'  # Token for class definition
    SWAG = 'SWAG'  # Token for referencing class instance
    THIS = 'THIS'  # Token for referencing current instance
    # ...