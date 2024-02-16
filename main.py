import sys
import global_variables

sys.dont_write_bytecode = True

from emoji_overlay import emoji_cam
from face_filters import face_filter_cam
from enum import Enum

class PhotoBoothOption(Enum):
    NONE = "0"
    EMOJI = "1"
    FACEFILTER = "2"

photo_booth_selection = PhotoBoothOption.NONE

# Entry point to the script
if __name__ == "__main__":

	# Photo Booth Instructions
	print("\n\nWelcome to the virtual photobooth for SST's 15th Anniversary!")
	print("This photobooth was developed by Tristan Chay and Klifton Cheng from the Class of 2024.")
	print("Pick a mode to get started! Press q at any time to return back to the selection menu, or press q now to quit the program.")

	while True:
		photo_booth_selection = PhotoBoothOption.NONE
		print("\n\nModes available:\n1: Emoji Mode\n- Express your facial expressions as emojis!\n2: Virtual Props Mode\nq: Quit the program\n")
		mode_selected = input()

		# Check for mode selectd
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

			wants_to_quit = False
			selecting_filters = True
			global_variables.filters_chosen = []

			while selecting_filters:

				# Virtual Props Instructions
				print("\nToggle Virtual Props:")
				print("1: Red Glasses Filter{}".format(" (SELECTED)" if global_variables.Filters.REDGLASSES in global_variables.filters_chosen else ""))
				print("2: Blue Glasses Filter{}".format(" (SELECTED)" if global_variables.Filters.BLUEGLASSES in global_variables.filters_chosen else ""))
				print("3: Gray Glasses Filter{}".format(" (SELECTED)" if global_variables.Filters.GRAYGLASSES in global_variables.filters_chosen else ""))
				print("4: Party Glasses Filter{}".format(" (SELECTED)" if global_variables.Filters.PARTYGLASSES in global_variables.filters_chosen else ""))
				print("5: Moustache Filter{}".format(" (SELECTED)" if global_variables.Filters.MOUSTACHE in global_variables.filters_chosen else ""))
				print("q: Go Back")
				print("When you are ready, just press enter.\n")

				filters_selected = input()

				# Assesses input
				if filters_selected == "":
					print("Loading Virutal Props...")
					selecting_filters = False
				elif filters_selected == "q":
					selecting_filters = False
					wants_to_quit = True
				else:
					# Selects or deselects filter
					try:
						global_variables.Filters(filters_selected)
						if global_variables.Filters(filters_selected) not in global_variables.filters_chosen:
							global_variables.filters_chosen.append(global_variables.Filters(filters_selected))
						else:
							global_variables.filters_chosen.remove(global_variables.Filters(filters_selected))
						filters_selected = ""
					except:
						print("Invalid option selected, try again.")

			# Starts face_filter_cam() if user does not want to quit
			if wants_to_quit == False:
				face_filter_cam()
