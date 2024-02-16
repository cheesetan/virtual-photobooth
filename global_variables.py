from enum import Enum

# Global variables and Enum
class Filters(Enum):
	REDGLASSES = "1"
	BLUEGLASSES = "2"
	GRAYGLASSES = "3"
	PARTYGLASSES = "4"
	MOUSTACHE = "5"
	
global filters_chosen
filters_chosen = []