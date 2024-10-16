"""
This file will contain all the code necessary for converting course material into a knowledge representation.
"""

class NoteToKR():
    """
    This class is responsible for taking input and converting it to the knowledge representation. The format for the knowledge representation can be 
    specified by the user and added later on.

    The user can do whatever they want with the KR afterwards.
    """

    def __init__(self,section_name=""):
        self.section_name = section_name
        self.input_text = ""                # Question: Should this be a list for various subsections
        self.input_images = ""          
        self.input_examples = []            # list of images?   # is this going to be in text format or images
    
    # need a __validate__input function?