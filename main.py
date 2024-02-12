import sys

sys.dont_write_bytecode = True

from emoji_overlay import emoji_cam
from face_filters import face_filter_cam
from enum import Enum

class PhotoBoothOption(Enum):
    NONE = "0"
    EMOJI = "1"
    FACEFILTER = "2"

global photo_booth_selection
photo_booth_selection = PhotoBoothOption.NONE


# Entry point to the script
if __name__ == "__main__":

	print("\n\nWelcome to the virtual photobooth for SST's 15th Anniversary!")
	print("This photobooth was developed by Tristan Chay and Klifton Cheng from the Class of 2024.")
	print("Pick a mode to get started! Press q at any time to return back to the selection menu, or press q now to quit the program.")

	while True:
		# if constants.photo_booth_selection == constants.PhotoBoothOption.NONE:
			photo_booth_selection = PhotoBoothOption.NONE
			print("\n\nModes available:\n1: Emoji Mode\n- Express your facial expressions as emojis!\n2: Filters Mode\nq: Quit the program\n")
			mode_selected = input()

			if mode_selected == "1" or mode_selected == "2":
				photo_booth_selection = PhotoBoothOption(mode_selected)
				mode_selected = ""
			elif mode_selected == "q":
				break
			else:
				print("Invalid option selected, try again.")

			if photo_booth_selection == PhotoBoothOption.EMOJI:
				emoji_cam()
			elif photo_booth_selection == PhotoBoothOption.FACEFILTER:
				face_filter_cam()
